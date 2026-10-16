import pandas as pd

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

params = [
    "BOD_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

for target in params:
    other = [x for x in params if x != target]

    print("\n", target)
    print("Missing:", df[target].isna().sum())

    for col in other:
        mask = df[target].isna() & df[col].notna()
        print(col, "available when target missing:", mask.sum())

    all_other = df[target].isna() & df[other].notna().all(axis=1)
    print("All other parameters available:", all_other.sum())