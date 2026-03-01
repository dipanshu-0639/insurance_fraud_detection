from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)


model = load("models/best_fraud_model.pkl")

incident_type_enc = load("models/encoders/incident_type_encoder.pkl")
collision_type_enc = load("models/encoders/collision_type_encoder.pkl")
incident_severity_enc = load("models/encoders/incident_severity_encoder.pkl")
authorities_enc = load("models/encoders/authorities_contacted_encoder.pkl")
property_damage_enc = load("models/encoders/property_damage_encoder.pkl")
police_report_enc = load("models/encoders/police_report_available_encoder.pkl")
insured_sex_enc = load("models/encoders/insured_sex_encoder.pkl")


@app.route("/")
def home():
    return render_template(
        "index.html",
        incident_types=incident_type_enc.classes_,
        collision_types=collision_type_enc.classes_,
        severities=incident_severity_enc.classes_,
        authorities=authorities_enc.classes_,
        property_damage=property_damage_enc.classes_,
        police_reports=police_report_enc.classes_,
        genders=insured_sex_enc.classes_,
    )


@app.route("/predict", methods=["POST"])
def predict():

    
    incident_type = request.form["incident_type"]
    collision_type = request.form["collision_type"]
    incident_severity = request.form["incident_severity"]
    authorities_contacted = request.form["authorities_contacted"]
    property_damage = request.form["property_damage"]
    police_report_available = request.form["police_report_available"]
    insured_sex = request.form["insured_sex"]

    policy_deductable = int(request.form["policy_deductable"])
    number_of_vehicles = int(request.form["number_of_vehicles"])
    bodily_injuries = int(request.form["bodily_injuries"])
    witnesses = int(request.form["witnesses"])
    injury_claim = int(request.form["injury_claim"])
    property_claim = int(request.form["property_claim"])
    vehicle_claim = int(request.form["vehicle_claim"])
    policy_annual_premium = int(request.form["policy_annual_premium"])

    
    incident_type = incident_type_enc.transform([incident_type])[0]
    collision_type = collision_type_enc.transform([collision_type])[0]
    incident_severity = incident_severity_enc.transform([incident_severity])[0]
    authorities_contacted = authorities_enc.transform([authorities_contacted])[0]
    property_damage = property_damage_enc.transform([property_damage])[0]
    police_report_available = police_report_enc.transform([police_report_available])[0]
    insured_sex = insured_sex_enc.transform([insured_sex])[0]

   
    features = [[
        policy_deductable,
        incident_type,
        collision_type,
        incident_severity,
        authorities_contacted,
        number_of_vehicles,
        property_damage,
        bodily_injuries,
        witnesses,
        police_report_available,
        injury_claim,
        property_claim,
        vehicle_claim,
        policy_annual_premium,
        insured_sex
    ]]

   
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        result = f"⚠️ Fraud Detected (Risk: {probability:.2f})"
    else:
        result = f"✅ Genuine Claim (Risk: {probability:.2f})"

    return render_template("index.html", prediction_text=result)



if __name__ == "__main__":
    app.run(debug=True)