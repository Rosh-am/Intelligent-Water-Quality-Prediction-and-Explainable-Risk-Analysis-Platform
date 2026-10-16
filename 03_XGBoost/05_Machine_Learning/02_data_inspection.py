import pandas as pd


baseline = pd.read_csv("NIEA_ML_Baseline.csv")
imputed = pd.read_csv("NIEA_ML_Imputed.csv")

datasets = {
    "Baseline": baseline,
    "Imputed": imputed
}

for name, df in datasets.items():

    print("\n" + "=" * 50)
    print(name.upper(), "DATASET")
    print("=" * 50)

    print("\nShape:", df.shape)

    duplicates = df.duplicated(
        subset=["StationCode", "Year"]
    ).sum()

    print("Duplicate Station-Year records:", duplicates)

    print("\nWQI Statistics:")
    print(df["CCME_WQI"].describe())

    print("\nWQI Category Distribution:")
    print(df["Category"].value_counts())

    features = [
        "BOD",
        "DO",
        "Ammonia_N",
        "Nitrite_N",
        "Nitrate_N",
        "pH"
    ]

    print("\nFeature Correlation with WQI:")
    print(
        df[features + ["CCME_WQI"]]
        .corr()["CCME_WQI"]
        .sort_values()
    )