import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


baseline_file = "../05_Machine_Learning/NIEA_ML_Baseline.csv"
imputed_file = "../05_Machine_Learning/NIEA_ML_Imputed.csv"


features = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

target = "CCME_WQI"

def run_dnn(file_path, dataset_name):

    print("\n" + "=" * 60)
    print("DNN -", dataset_name)
    print("=" * 60)

    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)

    data = df[features + [target]].copy()

    data = data.dropna(
        subset=features + [target]
    )

    print("Data used:", data.shape)

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

  
    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(X_train.shape[1],)),

        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dense(64, activation="relu"),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    print("\nDNN Model:")
    model.summary()

 

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        verbose=1
    )


    y_pred = model.predict(
        X_test,
        verbose=0
    ).flatten()



    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n" + "-" * 50)
    print(dataset_name, "DNN Results")
    print("-" * 50)

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))

  

    predictions = pd.DataFrame({
        "Actual_WQI": y_test.values,
        "Predicted_WQI": y_pred
    })

    predictions.to_csv(
        "dnn_" + dataset_name.lower() + "_predictions.csv",
        index=False
    )



    results = pd.DataFrame({
        "Model": ["DNN"],
        "Dataset": [dataset_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2": [r2]
    })

    results.to_csv(
        "dnn_" + dataset_name.lower() + "_metrics.csv",
        index=False
    )

    return results



baseline_results = run_dnn(
    baseline_file,
    "Baseline"
)


imputed_results = run_dnn(
    imputed_file,
    "Imputed"
)



final_results = pd.concat(
    [
        baseline_results,
        imputed_results
    ],
    ignore_index=True
)

final_results.to_csv(
    "dnn_model_comparison.csv",
    index=False
)


print("\n" + "=" * 60)
print("DNN FINAL RESULTS")
print("=" * 60)

print(final_results)