import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

params = [
    "BOD_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

results = []

for target in params:

    predictors = [x for x in params if x != target]

    data = df[predictors + ["Year", "Month", target]].copy()
    data = data.dropna()

    np.random.seed(42)

    mask = np.random.rand(len(data)) < 0.2

    train = data[~mask]
    test = data[mask]

    X_train = train[predictors + ["Year", "Month"]]
    y_train = train[target]

    X_test = test[predictors + ["Year", "Month"]]
    y_test = test[target]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append([
        target,
        len(train),
        len(test),
        mae,
        rmse,
        r2
    ])

    print("\n", target)
    print("Training:", len(train))
    print("Masked:", len(test))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

result = pd.DataFrame(
    results,
    columns=[
        "Parameter",
        "Training",
        "Masked",
        "MAE",
        "RMSE",
        "R2"
    ]
)

result.to_csv(
    "artificial_masking_results.csv",
    index=False
)

print("\nFinal Results")
print(result)