import pandas as pd

df = pd.read_csv("../02_Preprocessing/NIEA_core_WQI_candidate.csv")

params = ["BOD", "DO", "Ammonia_N", "Nitrite_N", "Nitrate_N", "pH"]

x = df.groupby(["StationCode", "Year"])[params].count()

available = (x > 0).sum(axis=1)

print(available.value_counts().sort_index())
print("\nTotal station-years:", len(available))