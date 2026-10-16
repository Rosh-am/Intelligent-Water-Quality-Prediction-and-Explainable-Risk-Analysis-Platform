import pandas as pd
import shap
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


def run_shap(file, name):

    print("\n" + "=" * 50)
    print(f"{name} SHAP ANALYSIS")
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

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title(f"SHAP Feature Impact - {name}")
    plt.tight_layout()

    plt.savefig(
        f"{name.lower()}_shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"{name} SHAP analysis completed.")


run_shap(
    "NIEA_ML_Baseline.csv",
    "Baseline"
)

run_shap(
    "NIEA_ML_Imputed.csv",
    "Imputed"
)

print("\nAll SHAP analysis completed successfully!")