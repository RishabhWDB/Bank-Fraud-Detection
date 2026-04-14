import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok = True)

np.random.seed(42)
n = 2000

transaction_id = np.arange(1 , n + 1)
amount = np.round(np.random.exponential(scale = 200 , size = n), 2)
is_fraud = np.random.choice([0,1], size = n, p = [0.95,0.05])

neg_indices = np.random.choice(n, size = 10, replace = False)
amount[neg_indices] = -np.abs(amount[neg_indices])

df = pd.DataFrame({
    "transactions_id" : transaction_id,
    "amount" : amount,
    "is_fraud" : is_fraud
})

df.to_csv("data/raw.csv", index = False)
print(f"Generated {len(df)} rows | Fraud rate: {df['is_fraud'].mean():.1%} | Negative amounts: {(df['amount'] < 0).sum()}")
