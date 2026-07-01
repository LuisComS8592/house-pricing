# House Pricing — End-to-End ML System

An end-to-end Machine Learning system that predicts residential real estate
prices, built as part of a Data Science capstone project (Projeto Integrador
IV and V) following the **CRISP-DM** methodology.

The project covers the full lifecycle: exploratory data analysis and model
selection in a notebook, a FastAPI service that serves the winning model with
SHAP-based explainability, a PostgreSQL database that logs every prediction,
and a Streamlit frontend for interactive use — all orchestrated with Docker
Compose.

**Team:** Laura de Freitas Sales, Luis Enrique Krulikowski

---

## Architecture

```
┌─────────────┐      HTTP       ┌─────────────┐      SQL      ┌─────────────┐
│  Streamlit  │ ───────────────▶│   FastAPI   │──────────────▶│  PostgreSQL │
│  (frontend) │◀─────────────── │    (api)    │◀────────────── │    (db)     │
└─────────────┘   JSON response └─────────────┘                └─────────────┘
                                       │
                                       ▼
                          CatBoost pipeline (.joblib)
                             + SHAP explainer
```

## Repository Structure

```
.
├── notebooks/
│   └── house_pricing_modeling.ipynb   # CRISP-DM: EDA, preprocessing, modeling
├── models/
│   └── pipeline_catboost_house_pricing.joblib   # Winning trained pipeline
├── data/
│   └── house_prices.csv               # Raw dataset (Kaggle)
├── api/                                # FastAPI service
│   ├── main.py
│   ├── database.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                           # Streamlit app
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   └── populate_db.py                  # Traffic simulation / smoke test
├── docker-compose.yml
├── .env.example
└── .gitattributes                      # Git LFS config
```

## Methodology (CRISP-DM)

The full modeling process — business understanding, data understanding, data
preparation, modeling (from linear regression to voting/blending/stacking
ensembles), and evaluation — is documented in
[`notebooks/house_pricing_modeling.ipynb`](notebooks/house_pricing_modeling.ipynb).
The `models/pipeline_catboost_house_pricing.joblib` file served by the API is
the production pipeline exported from that process (CatBoost with a custom
`DataPreprocessor` step, replicated in `api/main.py`).

## Getting Started

### Prerequisites

- Docker and Docker Compose
- [Git LFS](https://git-lfs.com/) (this repo stores the dataset, the trained
  model, and the SQLite artifact via LFS — run `git lfs install` once, then
  clone normally)

### 1. Configure environment variables

```bash
cp .env.example .env
# then edit .env with your own credentials
```

`.env` is git-ignored and must never be committed.

### 2. Run the full stack

```bash
docker compose up --build
```

- **Frontend (Streamlit):** http://localhost:8501
- **API (FastAPI docs):** http://localhost:8000/docs
- **PostgreSQL:** localhost:5432 (credentials from `.env`)

### 3. (Optional) Simulate traffic

With the stack running, populate the database with sample predictions:

```bash
cd scripts
pip install -r requirements.txt
python populate_db.py
```

## API Reference

### `POST /predict`

Request body (all fields required):

```json
{
  "date": "20260701T000000",
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "yr_built": 1990,
  "yr_renovated": 0,
  "zipcode": "98125",
  "lat": 47.7210,
  "long": -122.3190,
  "sqft_basement": 0
}
```

Response:

```json
{
  "status": "success",
  "record_id": 1,
  "processing_status": "approved",
  "estimated_price": 452380.55,
  "explainability": [
    {"feature": "grade", "effect": "Increased the value", "impact_weight": 0.182},
    ...
  ]
}
```

Properties predicted above US$ 1,000,000, or with more than 6 bedrooms or
4 bathrooms, are automatically flagged with `processing_status: "review"` for
manual review, instead of `"approved"`.

## Notes on the Refactor

This repository was reorganized from a flat, single-folder deployment into
the structure above. Along the way:

- **Security:** database credentials were moved out of `docker-compose.yml`
  and into a git-ignored `.env` file (`.env.example` documents the required
  variables).
- **Missing Dockerfile:** the API had no `Dockerfile` in the original
  deployment; one was created following the same pattern as the frontend's.
- **Model/data location:** the model and dataset are mounted as volumes
  (`models/`, `data/`) rather than baked into the API image, so they can be
  updated without rebuilding the container.
- **Translation:** all code, comments, and user-facing text were translated
  to English for portfolio consistency with the modeling notebook. API
  response field names changed accordingly (e.g. `preco_estimado` →
  `estimated_price`); if you have older client code pointing at the previous
  Portuguese field names or the `/prever` route, update it to match the
  reference above.
- **Large files:** `house_prices.csv`, the `.joblib` model, and the SQLite
  runtime file are tracked with Git LFS rather than committed as regular
  blobs. Note that the SQLite file (`house_pricing_db.db`) is a *runtime*
  artifact regenerated automatically by the API — you generally don't need
  to commit it at all; it's only tracked here because Postgres is the
  default in Docker and SQLite only kicks in as a local fallback.
