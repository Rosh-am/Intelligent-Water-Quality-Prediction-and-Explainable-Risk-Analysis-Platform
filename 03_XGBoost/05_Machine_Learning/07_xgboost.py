import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


features = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

target = "CCME_WQI"


def run_xgboost(file, name):

    df = pd.read_csv(file)

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


    model.fit(X_train, y_train)

 
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    results = pd.DataFrame({
        "Actual_WQI": y_test.values,
        "Predicted_WQI": predictions
    })

    print("\nFirst 10 predictions:")
    print(results.head(10))


run_xgboost(
    "NIEA_ML_Baseline.csv",
    "BASELINE XGBOOST"
)

run_xgboost(
    "NIEA_ML_Imputed.csv",
    "IMPUTED XGBOOST"
)