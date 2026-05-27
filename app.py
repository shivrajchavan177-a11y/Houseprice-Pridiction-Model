import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load Saved Model
# =========================

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
border-radius: 10px;
        font-size: 18px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        color: #155724;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# =========================
# App Title
# =========================

st.title("🏠 House Price Prediction System")

st.write("Predict house prices using Multiple Linear Regression")

st.divider()

# =========================
# User Inputs
# =========================

st.subheader("Enter House Details")

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

age = st.number_input(
    "House Age (Years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

area = st.number_input(
    "Area in Square Feet",
    min_value=500,
    max_value=10000,
    value=1500,
    step=100
)

st.divider()

# =========================
# Prediction Button
# =========================

if st.button("Predict House Price"):

    # Create DataFrame
    input_data = pd.DataFrame({
        'Bedrooms': [bedrooms],
        'Age': [age],
        'Area_SqFt': [area]
    })

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Show Result
    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted House Price:<br>
            ₹ {prediction:,.2f}
 </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Prediction Generated Successfully")

# =========================
# Sidebar Information
# =========================

st.sidebar.title("About Project")

st.sidebar.info(
    """
    This project uses:

    - Multiple Linear Regression
    - Scikit-learn
- Streamlit
    - Pandas

    Features Used:
    - Bedrooms
    - Age
    - Area_SqFt
    """
)

st.sidebar.write("Created using Machine Learning")
