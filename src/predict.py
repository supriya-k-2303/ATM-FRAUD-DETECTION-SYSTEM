# My model needs 29 inputs, but I only get 4, so I fill the missing ones with reasonable values 
# and add a simple rule to catch obviously fraud cases.
# FastAPI takes the input, sends it to the model and returns 
# whether its fraud along with a confidence score.

# ALL_FEATURES = [
#     'Per1','Per2','Per3','Per4','Per5','Per6','Per7','Per8','Per9',
#     'Dem1','Dem2','Dem3','Dem4','Dem5','Dem6','Dem7','Dem8','Dem9',
#     'Cred1','Cred2','Cred3','Cred4','Cred5','Cred6',
#     'Normalised_FNT','geo_score','qsets_normalized_tat',
#     'instance_scores','lambda_wt'
# ]

import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("model/best_fraud_pipeline.pkl")

# Expected columns
expected_cols = model.feature_names_in_

def predict_fraud(input_data):
    data = {}

    for col in expected_cols:
        if col in input_data:
            data[col] = input_data[col]
        else:
            # Fill missing features with realistic random values
            if "Per" in col:
                data[col] = np.random.uniform(0.3, 0.7)
            elif "Dem" in col:
                data[col] = np.random.uniform(0.2, 0.6)
            elif "Cred" in col:
                data[col] = np.random.uniform(0.5, 0.9)
            elif col == "Normalised_FNT":
                data[col] = np.random.uniform(0.3, 0.7)
            else:
                data[col] = 0

    df = pd.DataFrame([data])

    # Model prediction
    prediction = model.predict(df)[0]

    # Probability
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0][1]

    # Rule-based override 
    if input_data["geo_score"] > 0.8 and input_data["instance_scores"] > 0.8:
        return {
            "prediction": 1,
            "fraud_probability": 0.8,
            "risk_level": "HIGH"
        }

    # Risk level
    if proba is not None:
        if proba > 0.7:
            risk = "HIGH"
        elif proba > 0.3:
            risk = "MEDIUM"
        else:
            risk = "LOW"
    else:
        risk = "UNKNOWN"

    return {
        "prediction": int(prediction),
        "fraud_probability": float(proba) if proba else None,
        "risk_level": risk
    }