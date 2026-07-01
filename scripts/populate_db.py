"""Traffic simulation script.

Samples random properties from the raw dataset, sends them to the running
API, and reports how many were auto-approved versus sent for manual review.
Useful for smoke-testing the deployed stack end to end.
"""
import time
from datetime import datetime

import pandas as pd
import requests

# --- Configuration ---
CSV_FILE = "../data/house_prices.csv"
API_URL = "http://localhost:8000/predict"
NUM_HOUSES = 500

print(f"\U0001F680 Starting traffic simulation against the API ({NUM_HOUSES} houses)...")

# Translates text values to numbers, in case the CSV uses text labels
condition_map = {"Poor": 1, "Fair": 2, "Average": 3, "Good": 4, "Very Good": 5}

try:
    df = pd.read_csv(CSV_FILE)
    sample = df.sample(n=NUM_HOUSES, random_state=42)

    successes = 0
    reviews = 0

    for index, row in sample.iterrows():
        # --- Robust field handling ---
        # 1. Condition (translates text labels, keeps numbers as-is)
        condition_value = row["condition"]
        if isinstance(condition_value, str) and condition_value in condition_map:
            clean_condition = condition_map[condition_value]
        else:
            try:
                clean_condition = int(condition_value)
            except (TypeError, ValueError):
                clean_condition = 3  # default fallback

        # 2. Waterfront (can be 'Y'/'N' or 1/0)
        waterfront_value = row["waterfront"]
        if isinstance(waterfront_value, str):
            clean_waterfront = 1 if waterfront_value.upper() == "Y" else 0
        else:
            clean_waterfront = 1 if waterfront_value == 1 else 0

        payload = {
            "date": datetime.now().strftime("%Y%m%dT000000"),
            "bedrooms": float(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "sqft_living": float(row["sqft_living"]),
            "sqft_lot": float(row["sqft_lot"]),
            "floors": float(row["floors"]),
            "waterfront": clean_waterfront,
            # Extra NaN handling for secondary columns
            "view": int(row.get("view", 0)) if pd.notna(row.get("view")) else 0,
            "condition": clean_condition,
            "grade": int(row["grade"]) if pd.notna(row["grade"]) else 7,
            "yr_built": int(row["yr_built"]),
            "yr_renovated": (
                int(row["yr_renovated"]) if pd.notna(row["yr_renovated"]) else 0
            ),
            "zipcode": str(row["zipcode"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "sqft_basement": (
                float(row.get("sqft_basement", 0))
                if pd.notna(row.get("sqft_basement"))
                else 0
            ),
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            status = result["processing_status"]
            price = result["estimated_price"]
            record_id = result["record_id"]

            if status == "approved":
                print(
                    f"\u2705 ID {record_id:03d} | Approved | "
                    f"Estimated Price: US$ {price:,.2f}"
                )
                successes += 1
            else:
                print(
                    f"\u26A0\uFE0F ID {record_id:03d} | REVIEW   | "
                    f"Estimated Price: US$ {price:,.2f} (anomaly detected)"
                )
                reviews += 1
        else:
            print(f"\u274C Error on house index {index}: {response.text}")

        # Short pause to keep 500 requests from taking too long
        time.sleep(0.1)

    print("\n\U0001F389 Simulation finished successfully!")
    print(
        f"\U0001F4CA Summary: {successes} houses auto-approved and "
        f"{reviews} houses sent for manual review."
    )

except FileNotFoundError:
    print(f"\U0001F6A8 File '{CSV_FILE}' not found! Check the path relative to this script.")
except requests.exceptions.ConnectionError:
    print("\U0001F6A8 Connection failed! Is the API running (Docker or locally)?")
