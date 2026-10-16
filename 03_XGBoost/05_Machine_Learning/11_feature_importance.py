import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


features = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

target = "CCME_WQI"


def feature_importance(file, name):

    print("\n" + "=" * 50)
    print(f"{name} FEATURE IMPORTANCE")
    print("=" * 50)

    df = pd.read_csv(file)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

    model.fit(X_train, y_train)

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:\n")
    print(importance.to_string(index=False))

    importance.to_csv(
        f"{name.lower()}_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        importance["Feature"],
        importance["Importance"]
    )

    plt.xlabel("Water Quality Parameters")
    plt.ylabel("Feature Importance")
    plt.title(f"CatBoost Feature Importance - {name}")

    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        f"{name.lower()}_feature_importance.png",
        dpi=300
    )

    plt.show()


feature_importance(
    "NIEA_ML_Baseline.csv",
    "Baseline"
)

feature_importance(
    "NIEA_ML_Imputed.csv",
    "Imputed"
)

print("\nFeature importance analysis completed successfully!")