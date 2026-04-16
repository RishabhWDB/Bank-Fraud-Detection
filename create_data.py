import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok = True)

np.random.seed(42)
n = 2000

transaction_id = np.arange(1 , n + 1)
amount = np.round(np.random.exponential(scale = 200 , size = n), 2)
is_fraud = np.random.choice([0,1], size = n, p = [0.95,0.05])
num_transactions_24h = np.random.poisson(lam=2, size=n)
distance_from_home_km = np.round(np.random.exponential(scale=30, size=n), 2)
is_weekend = np.random.choice([0, 1], size=n, p=[5/7, 2/7])

neg_indices = np.random.choice(n, size = 10, replace = False)
amount[neg_indices] = -np.abs(amount[neg_indices])

df = pd.DataFrame({
    "transactions_id" : transaction_id,
    "amount" : amount,
    "num_transactions_24h": num_transactions_24h,
    "distance_from_home_km": distance_from_home_km,
    "is_weekend": is_weekend,
    "is_fraud" : is_fraud
})

df.to_csv("data/raw.csv", index = False)
print(f"Generated {len(df)} rows | Fraud rate: {df['is_fraud'].mean():.1%} | Negative amounts: {(df['amount'] < 0).sum()}")
