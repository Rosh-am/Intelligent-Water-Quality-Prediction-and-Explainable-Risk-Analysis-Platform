import pandas as pd

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

params = [
    "BOD_mg_l_",
    "DO_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

for target in params:
    df[target + "_Missing"] = df[target].isna().astype(int)

for target in params:
    missing = target + "_Missing"

    print("\n", target)
    print("Missing by year:")
    print(
        df.groupby("Year")[missing].mean().describe()
    )

    print("Missing by month:")
    print(
        df.groupby("Month")[missing].mean().round(3)
    )

print("\nStation-wise missingness")

for target in params:
    missing = target + "_Missing"
    station_missing = df.groupby("StationCode")[missing].mean()

    print(
        target,
        "Mean:", round(station_missing.mean(), 3),
        "Min:", round(station_missing.min(), 3),
        "Max:", round(station_missing.max(), 3)
    )