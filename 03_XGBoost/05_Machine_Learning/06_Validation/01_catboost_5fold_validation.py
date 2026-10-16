import os
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "06_Validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)



FEATURES = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

TARGET = "CCME_WQI"


datasets = {
    "Baseline": os.path.join(
        DATA_DIR,
        "NIEA_ML_Baseline.csv"
    ),

    "Imputed": os.path.join(
        DATA_DIR,
        "NIEA_ML_Imputed.csv"
    )
}



results = []

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for dataset_name, file_path in datasets.items():

    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    X = df[FEATURES]
    y = df[TARGET]

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold, (train_idx, test_idx) in enumerate(
        kf.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )

        model.fit(
            X_train,
            y_train
        )

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

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)

        print(
            f"Fold {fold}: "
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R²={r2:.4f}"
        )



    mean_mae = np.mean(fold_mae)
    std_mae = np.std(fold_mae, ddof=1)

    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse, ddof=1)

    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2, ddof=1)


    print("\n5-FOLD SUMMARY")
    print(
        f"MAE : {mean_mae:.4f} ± {std_mae:.4f}"
    )

    print(
        f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}"
    )

    print(
        f"R²  : {mean_r2:.4f} ± {std_r2:.4f}"
    )


    results.append({

        "Dataset": dataset_name,

        "Mean_MAE": mean_mae,
        "Std_MAE": std_mae,

        "Mean_RMSE": mean_rmse,
        "Std_RMSE": std_rmse,

        "Mean_R2": mean_r2,
        "Std_R2": std_r2
    })



results_df = pd.DataFrame(results)

output_file = os.path.join(
    OUTPUT_DIR,
    "catboost_5fold_validation_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nSaved to:")
print(output_file)