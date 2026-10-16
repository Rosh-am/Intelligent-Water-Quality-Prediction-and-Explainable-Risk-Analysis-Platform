import pandas as pd
import matplotlib.pyplot as plt

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


def prediction_analysis(file, name):

    print("\n" + "=" * 50)
    print(name)
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


    predictions = model.predict(X_test)


    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    prediction_df = pd.DataFrame({
        "Actual_WQI": y_test.values,
        "Predicted_WQI": predictions
    })

    prediction_df["Absolute_Error"] = (
        prediction_df["Actual_WQI"] -
        prediction_df["Predicted_WQI"]
    ).abs()

    print("\nFirst 10 Predictions:\n")
    print(prediction_df.head(10))

    prediction_df.to_csv(
        f"{name.lower().replace(' ', '_')}_predictions.csv",
        index=False
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        prediction_df["Actual_WQI"],
        prediction_df["Predicted_WQI"],
        alpha=0.5
    )

    minimum = min(
        prediction_df["Actual_WQI"].min(),
        prediction_df["Predicted_WQI"].min()
    )

    maximum = max(
        prediction_df["Actual_WQI"].max(),
        prediction_df["Predicted_WQI"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual WQI")
    plt.ylabel("Predicted WQI")
    plt.title(f"Actual vs Predicted WQI - {name}")

    plt.tight_layout()

    plt.savefig(
        f"{name.lower().replace(' ', '_')}_actual_vs_predicted.png",
        dpi=300
    )

    plt.show()



prediction_analysis(
    "NIEA_ML_Baseline.csv",
    "Baseline"
)

prediction_analysis(
    "NIEA_ML_Imputed.csv",
    "Imputed"
)

print("\nPrediction analysis completed successfully!")
