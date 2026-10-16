import pandas as pd
import matplotlib.pyplot as plt




ml_results = pd.DataFrame([
    ["Random Forest", "Baseline", 3.0245, 4.1346, 0.9231],
    ["Extra Trees", "Baseline", 3.0872, 4.1626, 0.9221],
    ["Gradient Boosting", "Baseline", 3.0791, 4.1093, 0.9240],
    ["XGBoost", "Baseline", 2.9187, 4.0063, 0.9278],
    ["CatBoost", "Baseline", 2.9632, 3.9849, 0.9286],

    ["Random Forest", "Imputed", 3.0870, 4.2095, 0.9204],
    ["Extra Trees", "Imputed", 3.1329, 4.1942, 0.9210],
    ["Gradient Boosting", "Imputed", 3.1068, 4.1384, 0.9231],
    ["XGBoost", "Imputed", 2.9579, 4.0368, 0.9268],
    ["CatBoost", "Imputed", 2.9799, 4.0024, 0.9280]
], columns=["Model", "Dataset", "MAE", "RMSE", "R2"])


dl_results = pd.DataFrame([
    ["DNN", "Baseline", 3.308817, 4.459447, 0.910541],
    ["DNN", "Imputed", 3.472902, 4.531214, 0.907751],
    ["TabNet", "Baseline", 4.855071, 6.338273, 0.819281],
    ["TabNet", "Imputed", 4.639898, 6.057183, 0.835156]
], columns=["Model", "Dataset", "MAE", "RMSE", "R2"])



all_results = pd.concat(
    [ml_results, dl_results],
    ignore_index=True
)

print("\n" + "=" * 70)
print("FINAL ML + DL MODEL COMPARISON")
print("=" * 70)
print(all_results.to_string(index=False))

all_results.to_csv(
    "final_ml_dl_model_comparison.csv",
    index=False
)



best_mae = all_results.loc[all_results["MAE"].idxmin()]
best_rmse = all_results.loc[all_results["RMSE"].idxmin()]
best_r2 = all_results.loc[all_results["R2"].idxmax()]

print("\n" + "=" * 70)
print("BEST MODEL BY MAE")
print("=" * 70)
print(best_mae)

print("\n" + "=" * 70)
print("BEST MODEL BY RMSE")
print("=" * 70)
print(best_rmse)

print("\n" + "=" * 70)
print("BEST MODEL BY R2")
print("=" * 70)
print(best_r2)


all_results["Model_Dataset"] = (
    all_results["Model"] + " - " + all_results["Dataset"]
)

plt.figure(figsize=(12, 6))

plt.bar(
    all_results["Model_Dataset"],
    all_results["RMSE"]
)

plt.xlabel("Model and Dataset")
plt.ylabel("RMSE")
plt.title("RMSE Comparison of Machine Learning and Deep Learning Models")
plt.xticks(rotation=75)
plt.tight_layout()

plt.savefig(
    "final_rmse_comparison.png",
    dpi=300
)

plt.show()



plt.figure(figsize=(12, 6))

plt.bar(
    all_results["Model_Dataset"],
    all_results["R2"]
)

plt.xlabel("Model and Dataset")
plt.ylabel("R²")
plt.title("R² Comparison of Machine Learning and Deep Learning Models")
plt.xticks(rotation=75)
plt.tight_layout()

plt.savefig(
    "final_r2_comparison.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(12, 6))

plt.bar(
    all_results["Model_Dataset"],
    all_results["MAE"]
)

plt.xlabel("Model and Dataset")
plt.ylabel("MAE")
plt.title("MAE Comparison of Machine Learning and Deep Learning Models")
plt.xticks(rotation=75)
plt.tight_layout()

plt.savefig(
    "final_mae_comparison.png",
    dpi=300
)

plt.show()