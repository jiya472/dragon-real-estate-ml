import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open("boston_real_estate_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 Boston House Price Predictor")
st.write("Enter the house features below:")

# Input fields
crim = st.number_input("CRIM (Crime rate)", 0.0, 100.0, step=0.1)
zn = st.number_input("ZN (Residential land > 25,000 sq.ft.)", 0.0, 100.0, step=0.5)
indus = st.number_input("INDUS (Non-retail business acres)", 0.0, 30.0)
chas = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
nox = st.number_input("NOX (Nitric oxides concentration)", 0.0, 1.0)
rm = st.number_input("RM (Avg rooms per dwelling)", 1.0, 10.0)
age = st.number_input("AGE (Old units %)", 0.0, 100.0)
dis = st.number_input("DIS (Distance to jobs)", 0.0, 15.0)
rad = st.slider("RAD (Highway access)", 1, 24)
tax = st.number_input("TAX (Property tax)", 100, 1000)
ptratio = st.number_input("PTRATIO (Pupil-teacher ratio)", 10.0, 30.0)
b = st.number_input("B (Black population index)", 0.0, 400.0)
lstat = st.number_input("LSTAT (Low-income %)", 0.0, 40.0)

# Predict button
if st.button("Predict House Price"):
    input_df = pd.DataFrame([{
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas, "NOX": nox,
        "RM": rm, "AGE": age, "DIS": dis, "RAD": rad, "TAX": tax,
        "PTRATIO": ptratio, "B": b, "LSTAT": lstat
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ${prediction * 1000:.2f}")
