import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Boston House Price Predictor", layout="centered")

# Load the model
model = joblib.load("boston_real_estate_model.pkl")

# App title and intro
st.title("🏠 Boston House Price Predictor")
st.markdown("Enter the housing features in the sidebar and click **Predict** to estimate the price.")

# Sidebar input
st.sidebar.header("🔧 Input Features")

crim = st.sidebar.number_input("CRIM (Crime rate)", 0.0, 100.0, step=0.1)
zn = st.sidebar.number_input("ZN (Residential land > 25,000 sq.ft.)", 0.0, 100.0, step=0.5)
indus = st.sidebar.number_input("INDUS (Non-retail business acres)", 0.0, 30.0)
chas = st.sidebar.selectbox("CHAS (Charles River dummy variable)", [0, 1])
nox = st.sidebar.number_input("NOX (Nitric oxides concentration)", 0.0, 1.0)
rm = st.sidebar.number_input("RM (Avg rooms per dwelling)", 1.0, 10.0)
age = st.sidebar.number_input("AGE (Old units %)", 0.0, 100.0)
dis = st.sidebar.number_input("DIS (Distance to jobs)", 0.0, 15.0)
rad = st.sidebar.slider("RAD (Highway access)", 1, 24)
tax = st.sidebar.number_input("TAX (Property tax)", 100, 1000)
ptratio = st.sidebar.number_input("PTRATIO (Pupil-teacher ratio)", 10.0, 30.0)
b = st.sidebar.number_input("B (Black population index)", 0.0, 400.0)
lstat = st.sidebar.number_input("LSTAT (Low-income %)", 0.0, 40.0)

# Predict
if st.button("Predict House Price"):
    input_df = pd.DataFrame([{
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas, "NOX": nox,
        "RM": rm, "AGE": age, "DIS": dis, "RAD": rad, "TAX": tax,
        "PTRATIO": ptratio, "B": b, "LSTAT": lstat
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ${prediction * 1000:.2f}")
