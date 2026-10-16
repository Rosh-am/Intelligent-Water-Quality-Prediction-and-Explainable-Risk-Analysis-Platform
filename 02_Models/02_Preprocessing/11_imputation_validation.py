import pandas as pd

df = pd.read_csv("NIEA_WQI_Imputed.csv")

params = [
    "BOD_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

for param in params:
    print("\n", param)

    print("Statistics:")
    print(
        df[param].describe()
    )

    print("Negative:", (df[param] < 0).sum())

    print(
        "Imputed:",
        df[param + "_imputed"].sum()
    )

print("\nRemaining missing values")

print(
    df[params].isna().sum()
)