"""Streamlit frontend for the House Pricing Predictor.

Collects property details from the user, sends them to the FastAPI backend,
and renders the predicted price alongside a SHAP explainability chart.
"""
import datetime
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

# Page configuration (must be the first Streamlit call)
st.set_page_config(
    page_title="House Pricing AI",
    page_icon="\U0001F3E1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Enlarges the main button's font */
    div.stButton > button:first-child {
        background-color: #2e6bc6;
        color: white;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
    }
    /* Hides Streamlit's default menu/footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1560518883-ce09059eeffa"
        "?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
        use_container_width=True,
    )
    st.title("\U0001F3E1 House Pricing AI")
    st.markdown(
        """
        **Welcome to the Intelligent Real Estate Pricing System.**

        This project uses **Machine Learning** combined with the
        **CRISP-DM** methodology to predict a property's market value
        based on its characteristics.
        """
    )
    st.divider()
    st.markdown("### \U0001F393 Capstone Project")
    st.markdown("**Program:** Data Science Technology")
    st.markdown("**Deployment status:** \U0001F7E2 Online")
    st.info(
        "Tip: use the tabs to navigate through the property's characteristics "
        "and click predict at the bottom of the page."
    )

# --- Main body ---
st.title("Real Estate Pricing Analysis")
st.markdown(
    "Fill in the form below with the property's characteristics. "
    "Our model will analyze the data in real time."
)

tab1, tab2, tab3 = st.tabs(
    ["\U0001F4CF Basic Characteristics", "\u2728 Quality and Condition",
     "\U0001F4CD Location and Other Details"]
)

with tab1:
    col1, col2, col3 = st.columns(3)
    bedrooms = col1.number_input(
        "Bedrooms", min_value=1, value=None, placeholder="e.g. 3", key="k_bed"
    )
    bathrooms = col2.number_input(
        "Bathrooms", min_value=0.0, value=None, step=1.0,
        placeholder="e.g. 2.0", key="k_bath",
    )
    floors = col3.number_input(
        "Floors", min_value=1.0, value=None, step=1.0,
        placeholder="e.g. 1.0", key="k_floors",
    )

    col4, col5 = st.columns(2)
    sqft_living = col4.number_input(
        "Living Area (sqft)", min_value=100, value=None,
        placeholder="e.g. 1500", key="k_sqft_liv",
    )
    sqft_lot = col5.number_input(
        "Lot Area (sqft)", min_value=100, value=None,
        placeholder="e.g. 5000", key="k_sqft_lot",
    )

with tab2:
    col1, col2 = st.columns(2)
    condition = col1.selectbox(
        "Property Condition", [1, 2, 3, 4, 5], index=None,
        placeholder="Select from 1 to 5...", key="k_cond",
    )
    grade = col2.number_input(
        "Construction/Design Grade", min_value=1, max_value=13, value=None,
        placeholder="e.g. 7 (1 to 13)", key="k_grade",
    )

    col3, col4, col5 = st.columns(3)
    waterfront = col3.selectbox(
        "Waterfront View?", ["No", "Yes"], index=None,
        placeholder="Select...", key="k_water",
    )
    view = col4.number_input(
        "View Quality (0 to 4)", min_value=0, max_value=4, value=None,
        placeholder="e.g. 0", key="k_view",
    )
    sqft_basement = col5.number_input(
        "Basement Area (0 if none)", min_value=0, value=None,
        placeholder="e.g. 0", key="k_base",
    )

with tab3:
    col1, col2, col3 = st.columns(3)
    yr_built = col1.number_input(
        "Year Built", min_value=1900, max_value=2026, value=None,
        placeholder="e.g. 1980", key="k_yr_b",
    )
    yr_renovated = col2.number_input(
        "Year Renovated (0 if never)", min_value=0, max_value=2026, value=None,
        placeholder="e.g. 0", key="k_yr_r",
    )
    zipcode = col3.text_input(
        "Zip Code", value=None, placeholder="e.g. 98125", key="k_zip"
    )

    col4, col5 = st.columns(2)
    lat = col4.number_input(
        "Latitude", value=None, format="%.4f",
        placeholder="e.g. 47.7210", key="k_lat",
    )
    long = col5.number_input(
        "Longitude", value=None, format="%.4f",
        placeholder="e.g. -122.3190", key="k_long",
    )

st.write("")  # spacing


def clear_form():
    """Resets every form field back to an empty state (runs before rerun)."""
    for key in st.session_state.keys():
        st.session_state[key] = None


# --- Action buttons ---
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    st.button("\U0001F504 Clear Data", use_container_width=True, on_click=clear_form)

with col_btn2:
    predict_clicked = st.button(
        "\U0001F680 Generate Price Prediction", type="primary",
        use_container_width=True,
    )

if predict_clicked:
    # 1. Validate that no field was left empty
    fields = [
        bedrooms, bathrooms, floors, sqft_living, sqft_lot, condition, grade,
        waterfront, view, sqft_basement, yr_built, yr_renovated, zipcode,
        lat, long,
    ]

    if any(field is None or field == "" for field in fields):
        st.error(
            "\u26A0\uFE0F Please fill in every field across the three tabs "
            "before generating a prediction!"
        )
    else:
        api_payload = {
            "date": datetime.datetime.now().strftime("%Y%m%dT000000"),
            "bedrooms": float(bedrooms),
            "bathrooms": float(bathrooms),
            "sqft_living": float(sqft_living),
            "sqft_lot": float(sqft_lot),
            "floors": float(floors),
            "waterfront": 1 if waterfront == "Yes" else 0,
            "view": int(view),
            "condition": int(condition),
            "grade": int(grade),
            "yr_built": int(yr_built),
            "yr_renovated": int(yr_renovated),
            "zipcode": str(zipcode),
            "lat": float(lat),
            "long": float(long),
            "sqft_basement": float(sqft_basement),
        }

        try:
            with st.spinner("The model is analyzing thousands of patterns..."):
                # Looks up API_URL from Docker; falls back to localhost
                api_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
                response = requests.post(api_url, json=api_payload)

            if response.status_code == 200:
                result = response.json()

                st.toast("Calculation finished successfully!", icon="\u2705")

                with st.container(border=True):
                    st.subheader("\U0001F4CA Valuation Result")

                    col_res1, col_res2, col_res3 = st.columns(3)
                    col_res1.metric(
                        label="Estimated Market Value",
                        value=f"US$ {result['estimated_price']:,.2f}",
                    )

                    status = result["processing_status"]
                    if status == "approved":
                        col_res2.success("\u2728 Status: Approved (High Confidence)")
                        st.balloons()
                    else:
                        col_res2.warning(
                            "\U0001F575\uFE0F\u200D\u2642\uFE0F Status: "
                            "Review Queue (Anomaly Detected)"
                        )

                    col_res3.metric(
                        label="Record ID (Database)",
                        value=f"#{result['record_id']}",
                    )

                st.write("---")
                st.subheader("\U0001F9E0 Model X-Ray (SHAP Explainability)")

                explainability_df = pd.DataFrame(result["explainability"])

                if not explainability_df.empty:
                    explainability_df["color"] = explainability_df["effect"].apply(
                        lambda effect: "#2ca02c" if "Increased" in effect else "#d62728"
                    )

                    chart = (
                        alt.Chart(explainability_df)
                        .mark_bar(cornerRadiusEnd=4, height=20)
                        .encode(
                            x=alt.X(
                                "impact_weight:Q",
                                title="Impact Strength (Absolute Value)",
                            ),
                            y=alt.Y("feature:N", sort="-x", title="Feature"),
                            color=alt.Color("color:N", scale=None),
                            tooltip=["feature", "effect", "impact_weight"],
                        )
                        .properties(height=300)
                        .configure_axis(labelFontSize=12, titleFontSize=14)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(
                        "No feature had a standout impact on this specific "
                        "prediction."
                    )

            else:
                st.error(f"API error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "\U0001F6A8 Could not connect to the API. Is Uvicorn running "
                "in the backend terminal?"
            )
