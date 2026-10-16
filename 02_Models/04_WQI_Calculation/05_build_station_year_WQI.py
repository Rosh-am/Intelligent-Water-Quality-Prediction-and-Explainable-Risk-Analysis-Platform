import pandas as pd
import numpy as np

df = pd.read_csv("../02_Preprocessing/NIEA_core_WQI_candidate.csv")

params = {
    "BOD": ("upper", 3.0),
    "Ammonia_N": ("upper", 0.05),
    "Nitrite_N": ("upper", 0.5),
    "Nitrate_N": ("upper", 0.5),
    "pH": ("range", (6.2, 7.3))
}

def failed(value, rule, limit):
    if rule == "upper":
        return value > limit
    return value < limit[0] or value > limit[1]

def excursion(value, rule, limit):
    if rule == "upper":
        return value / limit - 1
    if value < limit[0]:
        return limit[0] / value - 1
    return value / limit[1] - 1

results = []

for (station, year), group in df.groupby(["StationCode", "Year"]):

    if len(group) < 10:
        continue

    available = 0
    failed_parameters = 0
    failed_tests = 0
    total_tests = 0
    excursions = []

    for param, (rule, limit) in params.items():

        values = group[param].dropna()

        if len(values) == 0:
            continue

        available += 1

        for value in values:
            total_tests += 1

            if failed(value, rule, limit):
                failed_tests += 1
                excursions.append(excursion(value, rule, limit))

        if any(failed(value, rule, limit) for value in values):
            failed_parameters += 1

    if available < 4 or total_tests == 0:
        continue

    f1 = failed_parameters / available * 100
    f2 = failed_tests / total_tests * 100

    nse = sum(excursions) / total_tests if excursions else 0
    f3 = nse / (0.01 * nse + 0.01)

    wqi = 100 - np.sqrt(f1**2 + f2**2 + f3**2) / 1.732

    if wqi >= 95:
        category = "Excellent"
    elif wqi >= 80:
        category = "Good"
    elif wqi >= 65:
        category = "Fair"
    elif wqi >= 45:
        category = "Marginal"
    else:
        category = "Poor"

    results.append([
        station,
        year,
        len(group),
        available,
        f1,
        f2,
        f3,
        wqi,
        category
    ])

result = pd.DataFrame(results, columns=[
    "StationCode",
    "Year",
    "Observations",
    "Parameters",
    "F1",
    "F2",
    "F3",
    "CCME_WQI",
    "Category"
])

result.to_csv("NIEA_final_station_year_WQI.csv", index=False)

print(result.shape)
print(result.head())
print(result["Category"].value_counts())