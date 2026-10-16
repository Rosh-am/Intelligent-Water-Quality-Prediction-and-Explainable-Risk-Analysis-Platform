import pandas as pd

df = pd.read_csv("NIEA_final_station_year_WQI.csv")

print(df["CCME_WQI"].describe())

print("\nInvalid WQI:")
print((df["CCME_WQI"] < 0).sum())
print((df["CCME_WQI"] > 100).sum())

print("\nCategories:")
print(df["Category"].value_counts())

print("\nParameters:")
print(df["Parameters"].value_counts().sort_index())