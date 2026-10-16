import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

random_state = 42

os.makedirs("outputs", exist_ok=True)



def run_shap_analysis(dataset_name, file_name):

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS:", dataset_name)
    print("=" * 70)

    data = pd.read_csv(file_name)

    print("Original dataset shape:", data.shape)

    data = data.dropna(subset=features + [target]).copy()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state
    )


    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nCatBoost performance:")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))


    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_values = np.asarray(shap_values)

  

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_data = pd.DataFrame({
        "Feature": features,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance_data = importance_data.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    print("\nGlobal SHAP feature importance:")
    print(importance_data.to_string(index=False))

    importance_data.to_csv(
        f"outputs/shap_feature_importance_{dataset_name.lower()}.csv",
        index=False
    )


    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"Global SHAP Feature Importance - {dataset_name} Dataset"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/shap_bar_{dataset_name.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot - {dataset_name} Dataset"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/shap_summary_{dataset_name.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    for feature in features:

        shap.dependence_plot(
            feature,
            shap_values,
            X_test,
            show=False
        )

        plt.title(
            f"SHAP Dependence Plot: {feature} - {dataset_name}"
        )

        plt.tight_layout()

        safe_feature_name = feature.replace("/", "_")

        plt.savefig(
            f"outputs/shap_dependence_{dataset_name.lower()}_{safe_feature_name}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    prediction_data = X_test.copy()

    prediction_data["Actual_CCME_WQI"] = y_test.to_numpy()
    prediction_data["Predicted_CCME_WQI"] = predictions

    prediction_data.to_csv(
        f"outputs/shap_predictions_{dataset_name.lower()}.csv",
        index=False
    )

    shap_values_data = pd.DataFrame(
        shap_values,
        columns=[f"SHAP_{feature}" for feature in features]
    )

    shap_values_data.to_csv(
        f"outputs/shap_values_{dataset_name.lower()}.csv",
        index=False
    )

    return {
        "Model": "CatBoost",
        "Dataset": dataset_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }



baseline_result = run_shap_analysis(
    "Baseline",
    "../03_XGBoost/05_Machine_Learning/NIEA_ML_Baseline.csv"
)

imputed_result = run_shap_analysis(
    "Imputed",
    "../03_XGBoost/05_Machine_Learning/NIEA_ML_Imputed.csv"
)



shap_results = pd.DataFrame([
    baseline_result,
    imputed_result
])

print("\n" + "=" * 70)
print("SHAP CATBOOST PERFORMANCE RESULTS")
print("=" * 70)

print(shap_results.to_string(index=False))

shap_results.to_csv(
    "outputs/shap_catboost_model_results.csv",
    index=False
)

print("\nAll SHAP outputs were saved in the outputs folder.")