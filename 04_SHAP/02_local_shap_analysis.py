import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

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

random_state = 42

os.makedirs("outputs/local_explanations", exist_ok=True)



def run_local_shap(dataset_name, file_name):

    print("\n" + "=" * 70)
    print("LOCAL SHAP ANALYSIS:", dataset_name)
    print("=" * 70)

    data = pd.read_csv(file_name)

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


    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    expected_value = explainer.expected_value

    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]


    prediction_data = X_test.copy()

    prediction_data["Actual_CCME_WQI"] = y_test.to_numpy()
    prediction_data["Predicted_CCME_WQI"] = predictions

    prediction_data = prediction_data.reset_index(drop=True)

    lowest_index = prediction_data["Predicted_CCME_WQI"].idxmin()
    highest_index = prediction_data["Predicted_CCME_WQI"].idxmax()

    selected_records = {
        "Lowest_WQI": lowest_index,
        "Highest_WQI": highest_index
    }


    explanation_rows = []

    for record_type, record_index in selected_records.items():

        feature_values = X_test.reset_index(drop=True).iloc[record_index]
        record_shap_values = shap_values[record_index]

        predicted_wqi = predictions[record_index]
        actual_wqi = y_test.reset_index(drop=True).iloc[record_index]

        print("\n" + "-" * 70)
        print(record_type)
        print("-" * 70)
        print("Actual CCME-WQI:", round(actual_wqi, 4))
        print("Predicted CCME-WQI:", round(predicted_wqi, 4))

        record_table = pd.DataFrame({
            "Feature": features,
            "Actual_Value": feature_values.to_numpy(),
            "SHAP_Value": record_shap_values
        })

        record_table["Absolute_SHAP"] = np.abs(
            record_table["SHAP_Value"]
        )

        record_table = record_table.sort_values(
            by="Absolute_SHAP",
            ascending=False
        )

        print("\nParameter contributions:")
        print(record_table.to_string(index=False))

        record_table.to_csv(
            f"outputs/local_explanations/"
            f"{dataset_name.lower()}_{record_type.lower()}_contributions.csv",
            index=False
        )

   

        explanation = shap.Explanation(
            values=record_shap_values,
            base_values=expected_value,
            data=feature_values.to_numpy(),
            feature_names=features
        )

        shap.plots.waterfall(
            explanation,
            max_display=len(features),
            show=False
        )

        plt.title(
            f"Local SHAP Explanation - {dataset_name} - {record_type}"
        )

        plt.tight_layout()

        plt.savefig(
            f"outputs/local_explanations/"
            f"{dataset_name.lower()}_{record_type.lower()}_waterfall.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    

        for _, row in record_table.iterrows():

            explanation_rows.append({
                "Dataset": dataset_name,
                "Record_Type": record_type,
                "Actual_CCME_WQI": actual_wqi,
                "Predicted_CCME_WQI": predicted_wqi,
                "Feature": row["Feature"],
                "Actual_Value": row["Actual_Value"],
                "SHAP_Value": row["SHAP_Value"],
                "Absolute_SHAP": row["Absolute_SHAP"]
            })

    return explanation_rows



baseline_rows = run_local_shap(
    "Baseline",
    "../03_XGBoost/05_Machine_Learning/NIEA_ML_Baseline.csv"
)

imputed_rows = run_local_shap(
    "Imputed",
    "../03_XGBoost/05_Machine_Learning/NIEA_ML_Imputed.csv"
)



all_explanations = pd.DataFrame(
    baseline_rows + imputed_rows
)

all_explanations.to_csv(
    "outputs/local_explanations/all_local_shap_explanations.csv",
    index=False
)

print("\n" + "=" * 70)
print("LOCAL SHAP ANALYSIS COMPLETED")
print("=" * 70)

print(
    "Saved local explanations in: "
    "outputs/local_explanations"
)