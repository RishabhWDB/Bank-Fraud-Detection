import pandas as pd
import os
os.makedirs("data", exist_ok = True)

df = pd.read_csv("data/raw.csv")

df = df[df["amount"] > 0]

print(f"Rows before: {len(pd.read_csv('data/raw.csv'))} | Rows after: {len(df)} | {len(pd.read_csv('data/raw.csv')) - len(df)} lines removed")

df.to_csv("data/clean.csv", index = False)