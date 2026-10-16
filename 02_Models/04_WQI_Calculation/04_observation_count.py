import pandas as pd

df = pd.read_csv("../02_Preprocessing/NIEA_core_WQI_candidate.csv")

x = df.groupby(["StationCode", "Year"]).size()

print(x.describe())
print("\nStation-years:", len(x))
print("\nObservation count distribution:")
print(x.value_counts().sort_index().head(20))