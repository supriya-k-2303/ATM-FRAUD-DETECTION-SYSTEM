from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_fraud

app = FastAPI()

# Input schema
class Transaction(BaseModel):
    geo_score: float
    lambda_wt: float
    qsets_normalized_tat: float
    instance_scores: float

# Home route
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

# Health check 
@app.get("/health")
def health():
    return {"status": "API is running"}

# Prediction route
@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_fraud(transaction.dict())
    return result