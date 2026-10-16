import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

params = [
    "BOD_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

for param in params:
    df[param + "_imputed"] = 0

predictive = {
    "BOD_mg_l_": ["NH4_N_mg_l_", "NO2_N_mg_l_", "NO3_N_mg_l_", "PH_PHUNITS_", "Year", "Month"],
    "NH4_N_mg_l_": ["BOD_mg_l_", "NO2_N_mg_l_", "NO3_N_mg_l_", "PH_PHUNITS_", "Year", "Month"],
    "NO2_N_mg_l_": ["BOD_mg_l_", "NH4_N_mg_l_", "NO3_N_mg_l_", "PH_PHUNITS_", "Year", "Month"]
}

for target, features in predictive.items():

    train = df[df[target].notna()].dropna(subset=features)

    missing = df[target].isna()

    valid_missing = df.loc[missing].dropna(subset=features)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train[target]
    )

    if len(valid_missing) > 0:
        prediction = model.predict(valid_missing[features])

        df.loc[valid_missing.index, target] = prediction
        df.loc[valid_missing.index, target + "_imputed"] = 1

for target in ["NO3_N_mg_l_", "PH_PHUNITS_"]:

    df = df.sort_values(["StationCode", "Date"])

    previous = df.groupby("StationCode")[target].shift(1)
    next_value = df.groupby("StationCode")[target].shift(-1)

    mask = (
        df[target].isna()
        & previous.notna()
        & next_value.notna()
    )

    df.loc[mask, target] = (
        previous[mask] + next_value[mask]
    ) / 2

    df.loc[mask, target + "_imputed"] = 1

df.to_csv(
    "NIEA_WQI_Imputed.csv",
    index=False
)

print("Final records:", len(df))

for param in params:
    print(
        param,
        "Missing:", df[param].isna().sum(),
        "Imputed:", df[param + "_imputed"].sum()
    )