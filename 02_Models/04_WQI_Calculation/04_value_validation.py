import pandas as pd

df = pd.read_csv("../02_Preprocessing/NIEA_core_WQI_candidate.csv")

params = ["BOD", "DO", "Ammonia_N", "Nitrite_N", "Nitrate_N", "pH"]

for col in params:
    print("\n", col)
    print(df[col].describe())

print("\nNegative values")
print((df[params] < 0).sum())

print("\nDuplicates:", df.duplicated().sum())