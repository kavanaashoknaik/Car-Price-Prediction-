import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("car_price_model.pkl")
df = pd.read_csv("../dataset/car_data.csv")

df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -------------------- CUSTOM UI STYLING --------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit default UI */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}

    /* Full browser width container */
    .block-container {
        max-width: 1200px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* App background */
    .stApp {
        background-color: #0b1220;
        color: #e2e8f0;
    }

    /* Title */
    h1 {
        text-align: center;
        color: #f97316;
        font-weight: 900;
        letter-spacing: 1px;
    }

    /* Subtitle centered */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* Labels more visible */
    label {
        color: #e5e7eb !important;
        font-weight: 600;
    }

    /* Input boxes SMALLER */
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"] {
        max-width: 260px;
        margin: auto;
    }

    /* Input styling */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    /* Button */
    .stButton > button {
        background-color: #1f2937;
        color: #38bdf8;
        border-radius: 8px;
        border: 1px solid #334155;
        width: 100%;
        font-weight: bold;
    }

    .stButton > button:hover {
        border: 1px solid #f97316;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 Car Price Prediction ")
st.markdown("<p class='subtitle'>Enter car details below to predict selling price</p>", unsafe_allow_html=True)

# -------------------- INPUTS (2 COLUMN LAYOUT FOR FULL COVERAGE) --------------------
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Car Year", 2000, 2026, 2018)
    kms_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

with col2:
    present_price = st.number_input("Present Price (in lakhs)", 0.0, 100.0, 5.0)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Number of Owners", [0, 1, 2, 3])


# Convert inputs
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
trans_map = {"Manual": 0, "Automatic": 1}

fuel_type_value = fuel_map[fuel_type]
transmission_value = trans_map[transmission]

if st.button("Predict Price"):

    input_data = pd.DataFrame([[

        year,
        present_price,
        kms_driven,
        fuel_type_value,
        transmission_value,
        owner

    ]], columns=[

        "Year",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Transmission",
        "Owner"

    ])

    # Prediction
    prediction = model.predict(input_data)

    st.success(f"🚗 Estimated Price: ₹ {prediction[0]:.2f} Lakhs")

    # ---------------- BAR GRAPH ----------------

    fig, ax = plt.subplots(figsize=(6,4))

    labels = ["Present Price", "Predicted Selling Price"]

    values = [present_price, prediction[0]]

    colors = ["blue", "green"]

    ax.bar(labels, values, color=colors)

    ax.set_ylabel("Price (Lakhs)")
    ax.set_title("Car Price Comparison")

    st.pyplot(fig)