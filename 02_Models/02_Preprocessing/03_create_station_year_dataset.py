import pandas as pd

df = pd.read_csv("NIEA_core_WQI_candidate.csv")

station_year = (
    df.groupby(["StationCode", "Year"])
      .agg(
          Location=("Location", "first"),
          PrimaryBasin=("PrimaryBasin", "first"),
          GridReference=("GridReference", "first"),

          Observation_Count=("Year", "size"),

          BOD=("BOD", "median"),
          DO=("DO", "median"),
          Ammonia_N=("Ammonia_N", "median"),
          Nitrite_N=("Nitrite_N", "median"),
          Nitrate_N=("Nitrate_N", "median"),
          pH=("pH", "median")
      )
      .reset_index()
)

print("Total observations:", len(df))
print("Total Station-Year groups:", len(station_year))

station_year.to_csv(
    "NIEA_15502_Station_Year_Dataset.csv",
    index=False
)

print("Dataset saved successfully!")