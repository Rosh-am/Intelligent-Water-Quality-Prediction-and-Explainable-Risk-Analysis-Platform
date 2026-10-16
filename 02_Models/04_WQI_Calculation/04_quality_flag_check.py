import pandas as pd

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

cols = {
    "BOD": ("BOD_mg_l_", "Q_BOD"),
    "DO": ("DO_mg_l_", "Q_DO"),
    "Ammonia_N": ("NH4_N_mg_l_", "Q_NH4N"),
    "Nitrite_N": ("NO2_N_mg_l_", "Q_NO2N"),
    "Nitrate_N": ("NO3_N_mg_l_", "Q_NO3N"),
    "pH": ("PH_PHUNITS_", "Q_pH")
}

for name, (value, flag) in cols.items():
    x = df[df[value].notna()]

    print("\n", name)
    print(x[flag].value_counts(dropna=False).head(10))