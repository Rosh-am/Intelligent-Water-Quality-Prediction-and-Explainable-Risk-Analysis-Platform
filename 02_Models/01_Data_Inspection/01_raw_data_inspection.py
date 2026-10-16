import pandas as pd

df = pd.read_csv("NIEA_River_Water_Quality_Raw_1990_2024.csv", low_memory=False)

print(df.shape)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())

print("Stations:", df["StationCode"].nunique())

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print("Date range:", df["Date"].min(), "to", df["Date"].max())