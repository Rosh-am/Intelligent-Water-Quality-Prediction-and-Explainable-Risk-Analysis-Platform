import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


FEATURES = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

TARGET = "CCME_WQI"

DATASETS = {
    "Baseline": "NIEA_ML_Baseline.csv",
    "Imputed": "NIEA_ML_Imputed.csv"
}


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, rmse, r2


models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=0
    )
}


results = []


for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    for model_name, model in models.items():

        print("\nRunning:", model_name)

        mae, rmse, r2 = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("MAE :", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R²  :", round(r2, 4))

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })


results_df = pd.DataFrame(results)

results_df.to_csv(
    "master_ml_benchmark_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("FINAL MASTER ML BENCHMARK")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


baseline = results_df[
    results_df["Dataset"] == "Baseline"
]

imputed = results_df[
    results_df["Dataset"] == "Imputed"
]


plt.figure(figsize=(10, 5))

plt.plot(
    baseline["Model"],
    baseline["MAE"],
    marker="o",
    label="Baseline"
)

plt.plot(
    imputed["Model"],
    imputed["MAE"],
    marker="o",
    label="Imputed"
)

plt.title("Model Comparison using MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig("master_model_comparison_mae.png")
plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    baseline["Model"],
    baseline["RMSE"],
    marker="o",
    label="Baseline"
)

plt.plot(
    imputed["Model"],
    imputed["RMSE"],
    marker="o",
    label="Imputed"
)

plt.title("Model Comparison using RMSE")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig("master_model_comparison_rmse.png")
plt.close()


plt.figure(figsize=(10, 5))

plt.plot(
    baseline["Model"],
    baseline["R2"],
    marker="o",
    label="Baseline"
)

plt.plot(
    imputed["Model"],
    imputed["R2"],
    marker="o",
    label="Imputed"
)

plt.title("Model Comparison using R2 Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()
plt.savefig("master_model_comparison_r2.png")
plt.close()


print("\nBenchmark completed successfully.")
print("Results saved to: master_ml_benchmark_results.csv")