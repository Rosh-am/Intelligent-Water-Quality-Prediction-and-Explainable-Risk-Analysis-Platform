import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

print("Loading datasets...")

core = pd.read_csv(
    ROOT / "07_Models" / "02_Preprocessing" /
    "NIEA_core_WQI_candidate.csv"
)

baseline_wqi = pd.read_csv(
    ROOT / "07_Models" / "04_WQI_Calculation" /
    "NIEA_final_station_year_WQI.csv"
)

imputed_wqi = pd.read_csv(
    ROOT / "07_Models" / "04_WQI_New" /
    "NIEA_imputed_station_year_WQI.csv"
)

features = (
    core.groupby(["StationCode", "Year"])[
        ["BOD", "DO", "Ammonia_N",
         "Nitrite_N", "Nitrate_N", "pH"]
    ]
    .mean()
    .reset_index()
)

baseline_ml = features.merge(
    baseline_wqi[
        ["StationCode", "Year",
         "CCME_WQI", "Category"]
    ],
    on=["StationCode", "Year"],
    how="inner"
)

imputed_ml = features.merge(
    imputed_wqi[
        ["StationCode", "Year",
         "CCME_WQI", "Category"]
    ],
    on=["StationCode", "Year"],
    how="inner"
)

baseline_ml.to_csv(
    "NIEA_ML_Baseline.csv",
    index=False
)

imputed_ml.to_csv(
    "NIEA_ML_Imputed.csv",
    index=False
)

print("\nBaseline ML dataset:", baseline_ml.shape)
print("Imputed ML dataset:", imputed_ml.shape)

print("\nBaseline missing values:")
print(baseline_ml.isnull().sum())

print("\nImputed missing values:")
print(imputed_ml.isnull().sum())

print("\nData preparation completed successfully!")