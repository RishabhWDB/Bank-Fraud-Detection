from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

df = pd.read_csv("../data/clean.csv")
X = df[["amount", "num_transactions_24h", "distance_from_home_km", "is_weekend"]]
y = df["is_fraud"]
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

FEATURES = ["amount", "num_transactions_24h", "distance_from_home_km", "is_weekend"]

class Transaction(BaseModel):
    amount: float
    num_transactions_24h: int
    distance_from_home_km: float
    is_weekend: int

def get_risk_level(prob):
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "version": "1.0",
        "features": FEATURES
    }

@app.post("/predict")
def predict(transaction: Transaction):
    X_input = np.array([[
        transaction.amount,
        transaction.num_transactions_24h,
        transaction.distance_from_home_km,
        transaction.is_weekend
    ]])
    prob = model.predict_proba(X_input)[0][1]
    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": prob >= 0.5,
        "risk_level": get_risk_level(prob)
    }

@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    X_input = np.array([[
        t.amount,
        t.num_transactions_24h,
        t.distance_from_home_km,
        t.is_weekend
    ] for t in transactions])
    probs = model.predict_proba(X_input)[:, 1]
    return [
        {
            "fraud_probability": round(float(prob), 4),
            "is_fraud": prob >= 0.5,
            "risk_level": get_risk_level(prob)
        }
        for prob in probs
    ]