import pandas as pd

df = pd.read_csv("loan_data.csv")

print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)