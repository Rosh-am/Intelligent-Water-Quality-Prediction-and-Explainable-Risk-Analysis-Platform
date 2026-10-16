import os
import numpy as np
import pandas as pd
import streamlit as st
import shap
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "05_Recommendation_Engine")

FEATURES = [
    "BOD",
    "DO",
    "Ammonia_N",
    "Nitrite_N",
    "Nitrate_N",
    "pH"
]

GUIDELINES = {
    "BOD": {"upper": 3.0},
    "Ammonia_N": {"upper": 0.05},
    "Nitrite_N": {"upper": 0.5},
    "Nitrate_N": {"upper": 0.5},
    "pH": {"lower": 6.2, "upper": 7.3}
}

st.set_page_config(
    page_title="Water Quality Decision Support System",
    page_icon="💧",
    layout="wide"
)


@st.cache_resource
def load_model(dataset_name):

    if dataset_name == "Baseline":

        model_path = os.path.join(
            MODEL_DIR,
            "catboost_baseline_model.cbm"
        )

    else:

        model_path = os.path.join(
            MODEL_DIR,
            "catboost_imputed_model.cbm"
        )

    model = CatBoostRegressor()

    model.load_model(model_path)

    return model


def get_wqi_category(wqi):

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


def get_parameter_status(feature, value):

    guideline = GUIDELINES.get(feature)

    if guideline is None:
        return "Normal"

    if "upper" in guideline and value > guideline["upper"]:
        return "Exceeded"

    if "lower" in guideline and value < guideline["lower"]:
        return "Exceeded"

    return "Within guideline"


def get_pressure(feature):

    pressure_map = {

        "BOD":
        "Possible organic pollution or wastewater-related pressure.",

        "Ammonia_N":
        "Possible sewage, organic-waste, or nutrient-related pressure.",

        "Nitrite_N":
        "Possible nitrogen-cycle disturbance or wastewater-related pressure.",

        "Nitrate_N":
        "Possible nutrient runoff, agricultural, or wastewater-related pressure.",

        "pH":
        "Possible chemical or industrial influence.",

        "DO":
        "Possible oxygen-depletion or organic-pollution stress."
    }

    return pressure_map.get(
        feature,
        "Further investigation required."
    )


def get_recommendation(exceeded_parameters, category):

    if category == "Poor":
        priority = "High"

    elif category == "Marginal":
        priority = "High"

    elif len(exceeded_parameters) >= 2:
        priority = "High"

    elif category == "Fair":
        priority = "Medium"

    else:
        priority = "Low"

    if len(exceeded_parameters) == 0:

        recommendation = (
            "No selected guideline exceedance was detected. "
            "Continue routine water-quality monitoring and "
            "periodic assessment."
        )

        return priority, recommendation

    parameter_actions = {

        "BOD": {
            "pressure":
            "Possible organic pollution or wastewater-related pressure.",

            "action":
            (
                "Investigate relevant wastewater and organic-load sources, "
                "conduct targeted follow-up monitoring, and assess appropriate "
                "source-control or treatment measures."
            )
        },

        "Ammonia_N": {
            "pressure":
            "Possible sewage, organic-waste, or nutrient-related pressure.",

            "action":
            (
                "Investigate sewage, organic-waste, and nutrient inputs and "
                "assess appropriate treatment or source-control measures."
            )
        },

        "Nitrite_N": {
            "pressure":
            "Possible nitrogen-cycle disturbance or wastewater-related pressure.",

            "action":
            (
                "Investigate wastewater and nitrogen-related sources and assess "
                "whether nitrification-related treatment processes require attention."
            )
        },

        "Nitrate_N": {
            "pressure":
            "Possible nutrient runoff, agricultural, or wastewater-related pressure.",

            "action":
            (
                "Investigate agricultural runoff, nutrient inputs, and wastewater "
                "sources; consider appropriate nutrient-management or "
                "runoff-control measures."
            )
        },

        "pH": {
            "pressure":
            "Possible chemical or industrial influence.",

            "action":
            (
                "Investigate potential chemical or industrial influences and "
                "conduct site-specific source assessment before considering "
                "any controlled corrective action."
            )
        }
    }

    pressure_lines = []
    action_lines = []

    for parameter in exceeded_parameters:

        if parameter in parameter_actions:

            pressure = parameter_actions[parameter]["pressure"]
            action = parameter_actions[parameter]["action"]

            display_parameter = parameter.replace("_", "-")

            pressure_lines.append(
                f"- **{display_parameter}:** {pressure}"
            )

            action_lines.append(
                f"- **{display_parameter}:** {action}"
            )

    recommendation = ""

    if pressure_lines:

        recommendation += (
            "**Possible pressure interpretation**\n\n"
        )

        recommendation += (
            "\n".join(pressure_lines)
        )

        recommendation += "\n\n"

    if action_lines:

        recommendation += (
            "**Recommended actions**\n\n"
        )

        recommendation += (
            "\n".join(action_lines)
        )

        recommendation += "\n\n"

    recommendation += (
        "**Note:** These recommendations are decision-support suggestions "
        "based on measured parameters and selected guideline exceedances. "
        "They do not establish a confirmed pollution source or prescribe "
        "a specific remediation method."
    )

    return priority, recommendation


st.title(
    "💧 Water Quality Decision Support System"
)

st.markdown(
    """
    ### Explainable AI-Based Dynamic Water Quality Prediction

    Enter the measured water-quality parameters and click
    **Analyze Water Quality** to obtain a predicted CCME-WQI,
    model explanation, guideline status, and recommendation.
    """
)

st.sidebar.header("Model Selection")

dataset_name = st.sidebar.selectbox(
    "Choose Model Dataset",
    ["Baseline", "Imputed"]
)

model = load_model(dataset_name)

st.subheader(
    "Enter Water Quality Parameters"
)

col1, col2, col3 = st.columns(3)

with col1:

    BOD = st.number_input(
        "BOD",
        min_value=0.0,
        value=2.0,
        step=0.1,
        format="%.3f"
    )

    DO = st.number_input(
        "DO",
        min_value=0.0,
        value=7.0,
        step=0.1,
        format="%.3f"
    )

with col2:

    Ammonia_N = st.number_input(
        "Ammonia-N",
        min_value=0.0,
        value=0.04,
        step=0.01,
        format="%.4f"
    )

    Nitrite_N = st.number_input(
        "Nitrite-N",
        min_value=0.0,
        value=0.10,
        step=0.01,
        format="%.4f"
    )

with col3:

    Nitrate_N = st.number_input(
        "Nitrate-N",
        min_value=0.0,
        value=0.20,
        step=0.01,
        format="%.4f"
    )

    pH = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1,
        format="%.3f"
    )

st.divider()

analyze = st.button(
    "🔎 Analyze Water Quality",
    use_container_width=True
)

if analyze:

    input_data = pd.DataFrame(
        [[
            BOD,
            DO,
            Ammonia_N,
            Nitrite_N,
            Nitrate_N,
            pH
        ]],
        columns=FEATURES
    )

    predicted_wqi = float(
        model.predict(input_data)[0]
    )

    predicted_wqi = max(
        0.0,
        min(100.0, predicted_wqi)
    )

    category = get_wqi_category(
        predicted_wqi
    )

    st.subheader(
        "Prediction Result"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Predicted CCME-WQI",
            f"{predicted_wqi:.2f}"
        )

    with r2:

        st.metric(
            "Water Quality Category",
            category
        )

    with r3:

        st.metric(
            "Model",
            f"CatBoost ({dataset_name})"
        )

    st.subheader(
        "Input Parameter Summary"
    )

    summary_df = pd.DataFrame({

        "Parameter": FEATURES,

        "Input Value": [
            BOD,
            DO,
            Ammonia_N,
            Nitrite_N,
            Nitrate_N,
            pH
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Guideline Exceedance Analysis"
    )

    values = {

        "BOD": BOD,
        "DO": DO,
        "Ammonia_N": Ammonia_N,
        "Nitrite_N": Nitrite_N,
        "Nitrate_N": Nitrate_N,
        "pH": pH
    }

    status_rows = []
    exceeded_parameters = []

    for feature in FEATURES:

        status = get_parameter_status(
            feature,
            values[feature]
        )

        if status == "Exceeded":
            exceeded_parameters.append(feature)

        guideline = GUIDELINES.get(
            feature,
            {}
        )

        if (
            "lower" in guideline
            and
            "upper" in guideline
        ):

            guideline_text = (
                f"{guideline['lower']} – "
                f"{guideline['upper']}"
            )

        elif "upper" in guideline:

            guideline_text = (
                f"≤ {guideline['upper']}"
            )

        else:

            guideline_text = "—"

        status_rows.append({

            "Parameter": feature,

            "Measured Value":
            values[feature],

            "Guideline":
            guideline_text,

            "Status":
            status
        })

    status_df = pd.DataFrame(
        status_rows
    )

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Explainable AI — SHAP Analysis"
    )

    st.write(
        "SHAP values show how each input parameter contributes "
        "to the CatBoost prediction relative to the model's "
        "baseline prediction."
    )

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            input_data
        )

        shap_values = np.asarray(
            shap_values
        )

        if shap_values.ndim == 2:

            contributions = shap_values[0]

        else:

            contributions = shap_values

        shap_df = pd.DataFrame({

            "Parameter": FEATURES,

            "SHAP Contribution":
            contributions
        })

        shap_df["Absolute Contribution"] = (
            shap_df["SHAP Contribution"].abs()
        )

        shap_df = shap_df.sort_values(
            "Absolute Contribution",
            ascending=False
        )

        st.write(
            "**Positive SHAP value:** "
            "pushes the prediction upward."
        )

        st.write(
            "**Negative SHAP value:** "
            "pushes the prediction downward."
        )

        st.bar_chart(
            shap_df.set_index(
                "Parameter"
            )[
                "SHAP Contribution"
            ]
        )

        st.dataframe(
            shap_df[
                [
                    "Parameter",
                    "SHAP Contribution"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )

    st.subheader(
        "Decision Support Recommendation"
    )

    priority, recommendation = get_recommendation(
        exceeded_parameters,
        category
    )

    st.write(
        f"**Priority:** {priority}"
    )

    if exceeded_parameters:

        st.write(
            "**Parameters requiring attention:** "
            + ", ".join(
                parameter.replace("_", "-")
                for parameter in exceeded_parameters
            )
        )

    else:

        st.success(
            "No selected guideline exceedance detected."
        )

    st.markdown(
        recommendation
    )

    st.caption(
        "Note: The displayed CCME-WQI is a model prediction, "
        "not a directly measured WQI. SHAP values describe "
        "model attribution and should not be interpreted as "
        "proof of causal relationships."
    )