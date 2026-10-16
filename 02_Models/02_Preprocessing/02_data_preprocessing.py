import pandas as pd

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year

columns = [
    "StationCode",
    "Location",
    "PrimaryBasin",
    "GridReference",
    "Date",
    "Time",
    "Year",
    "BOD_mg_l_",
    "DO_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

data = df[columns].copy()

data = data.rename(columns={
    "BOD_mg_l_": "BOD",
    "DO_mg_l_": "DO",
    "NH4_N_mg_l_": "Ammonia_N",
    "NO2_N_mg_l_": "Nitrite_N",
    "NO3_N_mg_l_": "Nitrate_N",
    "PH_PHUNITS_": "pH"
})

data = data.dropna(
    subset=["BOD", "DO", "Ammonia_N", "Nitrite_N", "Nitrate_N", "pH"],
    how="all"
)

data.to_csv("NIEA_core_WQI_candidate.csv", index=False)

print(data.shape)
print(data.head())
print(data.isnull().sum())