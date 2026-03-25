import joblib

model = joblib.load("model/best_fraud_pipeline.pkl")

print(model.feature_names_in_)