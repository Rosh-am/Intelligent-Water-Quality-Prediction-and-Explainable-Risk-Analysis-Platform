import pandas as pd

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

checks = {
    "BOD_mg_l_": 100,
    "DO_mg_l_": 30,
    "NH4_N_mg_l_": 10,
    "NO2_N_mg_l_": 1,
    "NO3_N_mg_l_": 10,
    "PH_PHUNITS_": 10
}

for col, limit in checks.items():
    x = df[df[col] > limit]

    print("\n", col)
    print("Records:", len(x))
    print(x[["StationCode", "Date", col]].head(10))