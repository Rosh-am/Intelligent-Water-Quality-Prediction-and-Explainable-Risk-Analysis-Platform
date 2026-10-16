import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pytorch_tabnet.tab_model import TabNetRegressor


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


def run_tabnet(dataset_name, file_name):

    print("\n" + "=" * 60)
    print("TABNET DATASET:", dataset_name)
    print("=" * 60)

    data = pd.read_csv(file_name)

    print("Dataset shape:", data.shape)

    data = data.dropna(subset=features + [target]).copy()

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)

    y_train_array = y_train.to_numpy().reshape(-1, 1)
    y_test_array = y_test.to_numpy().reshape(-1, 1)

    model = TabNetRegressor(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
       optimizer_fn=torch.optim.Adam,
        seed=random_state,
        verbose=10
    )

    model.fit(
        X_train=X_train_scaled,
        y_train=y_train_array,
        eval_set=[
            (X_train_scaled, y_train_array),
            (X_test_scaled, y_test_array)
        ],
        eval_name=["train", "test"],
        eval_metric=["mae"],
        max_epochs=100,
        patience=15,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    predictions = model.predict(X_test_scaled).reshape(-1)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n" + "-" * 60)
    print("TABNET RESULTS")
    print("-" * 60)
    print("Dataset:", dataset_name)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    prediction_data = pd.DataFrame({
        "Actual_CCME_WQI": y_test.to_numpy(),
        "Predicted_CCME_WQI": predictions
    })

    prediction_data.to_csv(
        f"tabnet_{dataset_name.lower()}_predictions.csv",
        index=False
    )

    metrics_data = pd.DataFrame({
        "Model": ["TabNet"],
        "Dataset": [dataset_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2": [r2]
    })

    metrics_data.to_csv(
        f"tabnet_{dataset_name.lower()}_metrics.csv",
        index=False
    )

    return {
        "Model": "TabNet",
        "Dataset": dataset_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


baseline_result = run_tabnet(
    "Baseline",
    "../05_Machine_Learning/NIEA_ML_Baseline.csv"
)

imputed_result = run_tabnet(
    "Imputed",
    "../05_Machine_Learning/NIEA_ML_Imputed.csv"
)


final_results = pd.DataFrame([
    baseline_result,
    imputed_result
])

print("\n" + "=" * 60)
print("TABNET FINAL RESULTS")
print("=" * 60)

print(final_results)

final_results.to_csv(
    "tabnet_model_comparison.csv",
    index=False
)