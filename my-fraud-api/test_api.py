import requests
from tabulate import tabulate

transactions = [
    {"amount": 850, "num_transactions_24h": 9, "distance_from_home_km": 88, "is_weekend": 0},
    {"amount": 22,  "num_transactions_24h": 1, "distance_from_home_km": 3,  "is_weekend": 1},
    {"amount": 430, "num_transactions_24h": 4, "distance_from_home_km": 60, "is_weekend": 0},
    {"amount": 85,  "num_transactions_24h": 2, "distance_from_home_km": 12, "is_weekend": 1},
    {"amount": 1200,"num_transactions_24h": 12,"distance_from_home_km": 110,"is_weekend": 0},
]

results = []
for i, txn in enumerate(transactions, 1):
    r = requests.post("http://localhost:8000/predict", json=txn).json()
    results.append([
        i,
        f"${txn['amount']}",
        txn['num_transactions_24h'],
        f"{txn['distance_from_home_km']}km",
        f"{r['fraud_probability']*100:.1f}%",
        "FRAUD" if r['is_fraud'] else "legit"
    ])

print(tabulate(results, headers=["Txn", "Amount", "Txns/24h", "Distance", "Fraud%", "Verdict"]))