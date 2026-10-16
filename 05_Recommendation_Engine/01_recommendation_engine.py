import os
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASELINE_PATH = os.path.join(
    BASE_DIR,
    "..",
    "03_XGBoost",
    "05_Machine_Learning",
    "NIEA_ML_Baseline.csv"
)

IMPUTED_PATH = os.path.join(
    BASE_DIR,
    "..",
    "03_XGBoost",
    "05_Machine_Learning",
    "NIEA_ML_Imputed.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


FEATURES = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

TARGET = "CCME_WQI"



GUIDELINES = {
    "BOD": {
        "upper": 3.0
    },

    "Ammonia_N": {
        "upper": 0.05
    },

    "Nitrite_N": {
        "upper": 0.5
    },

    "Nitrate_N": {
        "upper": 0.5
    },

    "pH": {
        "lower": 6.2,
        "upper": 7.3
    }
}



def get_wqi_category(wqi):

    if pd.isna(wqi):
        return "Unknown"

    if wqi >= 95:
        return "Excellent"

    elif wqi >= 80:
        return "Good"

    elif wqi >= 65:
        return "Fair"

    elif wqi >= 45:
        return "Marginal"

    else:
        return "Poor"



def check_parameter(parameter, value):

    if pd.isna(value):
        return "Missing"

    guideline = GUIDELINES.get(parameter)

    if guideline is None:
        return "Within guideline"

    if "upper" in guideline and value > guideline["upper"]:
        return "Above guideline"

    if "lower" in guideline and value < guideline["lower"]:
        return "Below guideline"

    return "Within guideline"


PRESSURE_MAP = {

    "BOD":
        "Possible organic pollution or wastewater-related pressure",

    "Ammonia_N":
        "Possible sewage, organic-waste, or nutrient pollution pressure",

    "Nitrite_N":
        "Possible nitrogen-cycle disturbance or wastewater-related pressure",

    "Nitrate_N":
        "Possible nutrient runoff, agricultural, or wastewater-related pressure",

    "pH":
        "Possible chemical or industrial influence",

    "DO":
        "Possible oxygen-depletion or organic-pollution stress"
}


RECOMMENDATION_MAP = {

    "BOD":
        "Investigate possible organic or wastewater pollution sources and increase monitoring.",

    "Ammonia_N":
        "Investigate possible sewage or organic-waste inputs and increase monitoring of nitrogen pollution.",

    "Nitrite_N":
        "Investigate possible wastewater or nitrogen-cycle disturbances and monitor nitrogen compounds.",

    "Nitrate_N":
        "Investigate possible nutrient runoff or wastewater sources and strengthen nutrient monitoring.",

    "pH":
        "Investigate possible chemical or industrial influences and verify pH measurements.",

    "DO":
        "Increase monitoring for oxygen depletion and investigate possible organic-pollution sources."
}



def get_priority(wqi_category, number_of_exceedances):

    if wqi_category == "Poor":
        return "HIGH"

    if wqi_category == "Marginal":
        return "HIGH"

    if number_of_exceedances >= 2:
        return "HIGH"

    if wqi_category == "Fair":
        return "MEDIUM"

    return "LOW"


def train_catboost(X_train, y_train, model_path):

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    model.save_model(model_path)

    print(f"CatBoost model saved: {model_path}")

    return model



def generate_recommendations(file_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"Rows loaded: {len(df)}")



    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )



    model_df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    print(f"Rows available for modelling: {len(model_df)}")

    X = model_df[FEATURES]
    y = model_df[TARGET]

   

    model_path = os.path.join(
    BASE_DIR,
    f"catboost_{dataset_name.lower()}_model.cbm")

    model = train_catboost(
                   X,
                    y,
                     model_path
                         )


    predictions = model.predict(X)

    model_df["Predicted_CCME_WQI"] = predictions

    model_df["Predicted_Category"] = [
        get_wqi_category(value)
        for value in predictions
    ]

    model_df["Actual_Category"] = [
        get_wqi_category(value)
        for value in y
    ]


    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    parameter_status_columns = []
    negative_contributor_columns = []

    for parameter in FEATURES:

        status_column = f"{parameter}_Status"

        model_df[status_column] = [
            check_parameter(parameter, value)
            for value in model_df[parameter]
        ]

        parameter_status_columns.append(status_column)


    main_contributors = []
    contributor_values = []
    possible_pressures = []
    recommendations = []
    priorities = []

    for i in range(len(model_df)):

        row = model_df.iloc[i]


        exceeded_parameters = []

        for parameter in GUIDELINES:

            status = check_parameter(
                parameter,
                row[parameter]
            )

            if status in ["Above guideline", "Below guideline"]:
                exceeded_parameters.append(parameter)


        shap_row = shap_values[i]

        negative_features = []

        for j, parameter in enumerate(FEATURES):

            if shap_row[j] < 0:
                negative_features.append(
                    (
                        parameter,
                        shap_row[j]
                    )
                )

        negative_features.sort(
            key=lambda x: x[1]
        )


        if negative_features:

            main_parameter = negative_features[0][0]
            main_shap = negative_features[0][1]

        elif exceeded_parameters:

            main_parameter = exceeded_parameters[0]
            main_shap = np.nan

        else:

            main_parameter = "None"
            main_shap = np.nan


        if main_parameter in PRESSURE_MAP:

            pressure = PRESSURE_MAP[main_parameter]

        else:

            pressure = "No major pollution pressure identified by the rule-based system"


        if main_parameter in RECOMMENDATION_MAP:

            recommendation = RECOMMENDATION_MAP[main_parameter]

        else:

            recommendation = (
                "Continue routine monitoring and maintain current "
                "water-quality surveillance."
            )


        priority = get_priority(
            row["Predicted_Category"],
            len(exceeded_parameters)
        )

        main_contributors.append(main_parameter)
        contributor_values.append(main_shap)
        possible_pressures.append(pressure)
        recommendations.append(recommendation)
        priorities.append(priority)

  

    model_df["Main_SHAP_Contributor"] = main_contributors

    model_df["Main_SHAP_Value"] = contributor_values

    model_df["Possible_Pollution_Pressure"] = possible_pressures

    model_df["Recommendation"] = recommendations

    model_df["Monitoring_Priority"] = priorities


    output_file = os.path.join(
        OUTPUT_DIR,
        f"recommendations_{dataset_name.lower()}.csv"
    )

    model_df.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved: {output_file}")


    print("\nPredicted WQI category distribution:")

    print(
        model_df["Predicted_Category"]
        .value_counts()
    )

    print("\nMonitoring priority distribution:")

    print(
        model_df["Monitoring_Priority"]
        .value_counts()
    )

    return model_df



baseline_results = generate_recommendations(
    BASELINE_PATH,
    "baseline"
)



imputed_results = generate_recommendations(
    IMPUTED_PATH,
    "imputed"
)



print("\n" + "=" * 70)
print("RECOMMENDATION ENGINE COMPLETED")
print("=" * 70)

print("\nOutputs created inside:")
print(OUTPUT_DIR)