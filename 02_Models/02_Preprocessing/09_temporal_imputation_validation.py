import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    "../01_Data_Inspection/NIEA_River_Water_Quality_Raw_1990_2024.csv",
    low_memory=False
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

params = [
    "BOD_mg_l_",
    "NH4_N_mg_l_",
    "NO2_N_mg_l_",
    "NO3_N_mg_l_",
    "PH_PHUNITS_"
]

results = []

for param in params:

    data = df[["StationCode", "Date", param]].dropna().copy()
    data = data.sort_values(["StationCode", "Date"])

    data["Previous"] = data.groupby("StationCode")[param].shift(1)
    data["Next"] = data.groupby("StationCode")[param].shift(-1)

    test = data.dropna(subset=["Previous", "Next"]).copy()

    test["Prediction"] = (
        test["Previous"] + test["Next"]
    ) / 2

    mae = mean_absolute_error(
        test[param],
        test["Prediction"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test[param],
            test["Prediction"]
        )
    )

    r2 = r2_score(
        test[param],
        test["Prediction"]
    )

    results.append([
        param,
        len(test),
        mae,
        rmse,
        r2
    ])

    print("\n", param)
    print("Validation records:", len(test))
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

result = pd.DataFrame(
    results,
    columns=["Parameter", "Validation", "MAE", "RMSE", "R2"]
)

result.to_csv(
    "temporal_imputation_results.csv",
    index=False
)

print("\nFinal Results")
print(result)