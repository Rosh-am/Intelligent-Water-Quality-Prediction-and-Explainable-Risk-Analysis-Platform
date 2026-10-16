import pandas as pd
from sklearn.model_selection import train_test_split


baseline = pd.read_csv("NIEA_ML_Baseline.csv")
imputed = pd.read_csv("NIEA_ML_Imputed.csv")


features = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

target = "CCME_WQI"


def prepare_data(df, name):

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test

baseline_data = prepare_data(
    baseline,
    "BASELINE DATASET"
)


imputed_data = prepare_data(
    imputed,
    "IMPUTED DATASET"
)