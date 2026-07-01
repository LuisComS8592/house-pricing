"""FastAPI service that serves the House Pricing regression model.

Exposes a single `/predict` endpoint that returns an estimated price plus a
SHAP-based explanation of the top features driving that prediction. Every
request/response pair is persisted to the database for auditing.
"""
import os

import __main__
import joblib
import numpy as np
import pandas as pd
import shap
from database import PredictionRecord, SessionLocal
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy.orm import Session


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer replicating the notebook's feature
    engineering steps (zip code target encoding, house age, etc.) so the
    trained pipeline can be unpickled and reused at inference time.
    """

    def __init__(self):
        self.zip_median_map_ = {}
        self.global_median_ = 0

    def fit(self, X, y=None):
        if y is not None and "zipcode" in X.columns:
            y_real = np.expm1(y)
            temp_df = pd.DataFrame(
                {"zipcode": X["zipcode"].values, "price": y_real.values}
            )
            self.zip_median_map_ = (
                temp_df.groupby("zipcode")["price"].median().to_dict()
            )
            self.global_median_ = y_real.median()
        return self

    def transform(self, X, y=None):
        df = X.copy()
        if "zipcode" in df.columns:
            df["zipcode_rank"] = (
                df["zipcode"].map(self.zip_median_map_).fillna(self.global_median_)
            )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["sale_year"] = df["date"].dt.year
        if "yr_built" in df.columns and "sale_year" in df.columns:
            df["house_age"] = df["sale_year"] - df["yr_built"]
            df["house_age"] = df["house_age"].clip(lower=0)
        if "yr_renovated" in df.columns:
            df["was_renovated"] = df["yr_renovated"].apply(
                lambda year: 1 if year > 0 else 0
            )
        if "sqft_basement" in df.columns:
            df["has_basement"] = df["sqft_basement"].apply(
                lambda area: 1 if area > 0 else 0
            )
        if "waterfront" in df.columns and df["waterfront"].dtype == object:
            df["waterfront"] = df["waterfront"].map({"N": 0, "Y": 1}).fillna(0)
        if "condition" in df.columns and df["condition"].dtype == object:
            condition_map = {
                "Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5,
            }
            df["condition"] = df["condition"].map(condition_map).fillna(3)

        columns_to_drop = [
            "id", "date", "yr_built", "yr_renovated",
            "sqft_basement", "sqft_above", "sqft_living15", "sqft_lot15",
            "zipcode",
        ]
        existing_drops = [c for c in columns_to_drop if c in df.columns]
        return df.drop(columns=existing_drops)


app = FastAPI(title="House Pricing Predictor API")


def get_db():
    """Yields a database session and guarantees it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# The trained pipeline was pickled while DataPreprocessor lived in the
# training script's __main__ module. Registering the class here lets
# joblib.load() resolve it correctly at inference time.
__main__.DataPreprocessor = DataPreprocessor

# Model loading. MODEL_PATH can be overridden via environment variable
# (Docker Compose mounts the models/ folder and sets this for us).
MODEL_PATH = os.getenv(
    "MODEL_PATH", "../models/pipeline_catboost_house_pricing.joblib"
)
pricing_pipeline = joblib.load(MODEL_PATH)

# Extracts the underlying CatBoost model from the pipeline for SHAP
catboost_model = pricing_pipeline.named_steps["model"]
explainer = shap.TreeExplainer(catboost_model)


class HouseInput(BaseModel):
    """Request payload for a single property to be priced."""

    date: str
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_basement: float


@app.post("/predict")
def predict_price(house: HouseInput, db: Session = Depends(get_db)):
    """Predicts the sale price for a property and explains the result."""
    try:
        input_df = pd.DataFrame([house.model_dump()])

        # Predict in log scale, then revert to the real dollar scale
        predicted_price_log = pricing_pipeline.predict(input_df)[0]
        predicted_price = np.expm1(predicted_price_log)

        # SHAP explainability: break the pipeline apart to get feature names
        preprocessor = pricing_pipeline.named_steps["preprocessor"]
        scaler = pricing_pipeline.named_steps["scaler"]

        processed_df = preprocessor.transform(input_df)
        feature_names = processed_df.columns.tolist()

        # Scale the features, since CatBoost was trained on scaled data
        scaled_features = scaler.transform(processed_df)

        shap_values = explainer.shap_values(scaled_features)[0]

        # Ranked list of the features that most influenced the price
        explanations = []
        for feature_name, shap_value in zip(feature_names, shap_values):
            # Only keep features with a meaningful impact (> 0.01 in log scale)
            if abs(shap_value) > 0.01:
                explanations.append({
                    "feature": feature_name,
                    "effect": (
                        "Increased the value"
                        if shap_value > 0
                        else "Decreased the value"
                    ),
                    "impact_weight": round(abs(shap_value), 3),
                })

        explanations = sorted(
            explanations, key=lambda item: item["impact_weight"], reverse=True
        )

        # Simple business rule to flag properties for manual review
        processing_status = "approved"
        if predicted_price > 1_000_000:  # very high-value property
            processing_status = "review"
        elif house.bedrooms > 6:  # unusually large property
            processing_status = "review"
        elif house.bathrooms > 4:
            processing_status = "review"

        # Persist the request/response pair for auditing
        new_record = PredictionRecord(
            date=house.date,
            bedrooms=house.bedrooms,
            bathrooms=house.bathrooms,
            sqft_living=house.sqft_living,
            sqft_lot=house.sqft_lot,
            floors=house.floors,
            waterfront=house.waterfront,
            view=house.view,
            condition=house.condition,
            grade=house.grade,
            yr_built=house.yr_built,
            yr_renovated=house.yr_renovated,
            zipcode=house.zipcode,
            lat=house.lat,
            long=house.long,
            sqft_basement=house.sqft_basement,
            estimated_price=float(predicted_price),
            status=processing_status,
        )

        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        return {
            "status": "success",
            "record_id": new_record.id,
            "processing_status": processing_status,
            "estimated_price": round(float(predicted_price), 2),
            "explainability": explanations[:5],
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}")
