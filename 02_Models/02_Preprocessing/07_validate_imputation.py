import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
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

features = params.copy()
results = []

for target in params:

    predictors = [x for x in features if x != target]

    data = df[predictors + ["Year", "Month", target]].dropna()

    if len(data) < 1000:
        print("\n", target)
        print("Insufficient complete records:", len(data))
        continue

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

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
    print("Validation:", len(test))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

result_df = pd.DataFrame(
    results,
    columns=[
        "Parameter",
        "Training",
        "Validation",
        "MAE",
        "RMSE",
        "R2"
    ]
)

result_df.to_csv(
    "imputation_validation_results.csv",
    index=False
)

print("\nFinal comparison")
print(result_df)