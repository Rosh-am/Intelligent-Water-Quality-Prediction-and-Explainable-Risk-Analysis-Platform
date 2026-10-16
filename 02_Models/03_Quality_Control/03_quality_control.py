import pandas as pd

df = pd.read_csv("../02_Preprocessing/NIEA_core_WQI_candidate.csv")

params = ["BOD", "DO", "Ammonia_N", "Nitrite_N", "Nitrate_N", "pH"]

report = []

for col in params:
    report.append([
        col,
        df[col].notna().sum(),
        df[col].isna().sum(),
        round(df[col].isna().mean() * 100, 2),
        df[col].min(),
        df[col].max()
    ])

report = pd.DataFrame(report, columns=[
    "Parameter",
    "Available",
    "Missing",
    "Missing_Percent",
    "Minimum",
    "Maximum"
])

print(report)

report.to_csv("NIEA_quality_control_report.csv", index=False)