import pandas as pd

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

params = [
    "BOD_mg_l_",
    "DO_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

for col in params:
    print("\n", col)
    print("Missing:", df[col].isna().sum())
    print("Available:", df[col].notna().sum())
    print("Missing %:", round(df[col].isna().mean() * 100, 2))

print("\nMissingness by year")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

for col in params:
    missing_by_year = df.groupby("Year")[col].apply(lambda x: x.isna().mean() * 100)
    print("\n", col)
    print(missing_by_year.describe())