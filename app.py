import streamlit as st
import pandas as pd
from joblib import load
import os

# ===============================
# Load Model & Encoders
# ===============================

MODEL_PATH = "models/best_fraud_model.pkl"
ENCODER_PATH = "models/encoders/"

model = load(MODEL_PATH)

incident_type_enc = load(os.path.join(ENCODER_PATH, "incident_type_encoder.pkl"))
collision_type_enc = load(os.path.join(ENCODER_PATH, "collision_type_encoder.pkl"))
incident_severity_enc = load(os.path.join(ENCODER_PATH, "incident_severity_encoder.pkl"))
authorities_enc = load(os.path.join(ENCODER_PATH, "authorities_contacted_encoder.pkl"))
property_damage_enc = load(os.path.join(ENCODER_PATH, "property_damage_encoder.pkl"))
police_report_enc = load(os.path.join(ENCODER_PATH, "police_report_available_encoder.pkl"))
insured_sex_enc = load(os.path.join(ENCODER_PATH, "insured_sex_encoder.pkl"))

# ===============================
# Page UI
# ===============================

st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")
st.title("🚨 Insurance Fraud Detection")
st.write("Fill the details below to check if the claim is fraudulent.")

# ===============================
# User Inputs
# ===============================

incident_type = st.selectbox(
    "Incident Type", incident_type_enc.classes_
)

collision_type = st.selectbox(
    "Collision Type", collision_type_enc.classes_
)

incident_severity = st.selectbox(
    "Incident Severity", incident_severity_enc.classes_
)

authorities_contacted = st.selectbox(
    "Authorities Contacted", authorities_enc.classes_
)

property_damage = st.selectbox(
    "Property Damage", property_damage_enc.classes_
)

police_report_available = st.selectbox(
    "Police Report Available", police_report_enc.classes_
)

insured_sex = st.selectbox(
    "Insured Sex", insured_sex_enc.classes_
)


number_of_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=0)
bodily_injuries = st.number_input("Bodily Injuries", min_value=0)
witnesses = st.number_input("Witnesses", min_value=0)

injury_claim = st.number_input("Injury Claim Amount", min_value=0.0)
property_claim = st.number_input("Property Claim Amount", min_value=0.0)
vehicle_claim = st.number_input("Vehicle Claim Amount", min_value=0.0)
policy_annual_premium = st.number_input("Policy Annual Premium", min_value=0.0)

# ===============================
# Prediction Button
# ===============================

if st.button("Predict Fraud"):

    # Encode categorical features
    encoded_features = [
        
        incident_type_enc.transform([incident_type])[0],
        collision_type_enc.transform([collision_type])[0],
        incident_severity_enc.transform([incident_severity])[0],
        authorities_enc.transform([authorities_contacted])[0],
        number_of_vehicles_involved,
        property_damage_enc.transform([property_damage])[0],
        bodily_injuries,
        witnesses,
        police_report_enc.transform([police_report_available])[0],
        injury_claim,
        property_claim,
        vehicle_claim,
        policy_annual_premium,
        insured_sex_enc.transform([insured_sex])[0],
    ]

    prediction = model.predict([encoded_features])[0]
    proba=model.predict_proba([encoded_features])[0][1]
    st.write(f"Fraud Probability: {proba:.2f}")
    
    if proba >0.7:
        st.error("⚠️ High Risk of Fraudulent Claim Detected!")
    elif proba >0.4:
        st.warning("⚠️ Moderate Risk of Fraudulent Claim Detected!")
    else:
        st.success("✅ Low Risk of Fraudulent Claim Detected.")

    if prediction == 1:
        st.error("⚠️ Fraudulent Claim Detected!")
    else:
        st.success("✅ Claim is Legitimate.")