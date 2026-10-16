<div align="center">

# 💧 Water Quality Risk Assessment
### Explainable Machine Learning for Water Quality Prediction & Decision Support

**From historical river measurements to interpretable water-quality estimates and actionable monitoring guidance.**

<p>
  <a href="https://water-quality-risk-assessment.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Launch_App-16A34A?style=for-the-badge" alt="Launch Live Demo">
  </a>
  <a href="https://github.com/Rosh-am/Intelligent-Water-Quality-Prediction-and-Explainable-Risk-Analysis-Platform">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Project-Capstone-0EA5E9?style=for-the-badge" alt="Capstone project">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ML-CatBoost%20%7C%20XGBoost-16A34A?style=for-the-badge" alt="Machine learning">
  <img src="https://img.shields.io/badge/Explainability-SHAP-7C3AED?style=for-the-badge" alt="Explainable AI">
  <img src="https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

**A data-science capstone project** combining data-quality assessment, missing-value handling, Water Quality Index (WQI) modelling, explainable AI, and an interactive Streamlit decision-support dashboard.

### 🌐 Try the application

# [💧 Open Water Quality Decision Support System →](https://water-quality-risk-assessment.streamlit.app/)

*Explore water-quality predictions, SHAP-based explanations, and decision-support recommendations.*

</div>

---

## 🌊 The project in 30 seconds

Water-quality records can be incomplete, difficult to compare across stations and years, and hard to translate into practical follow-up actions. This project builds an end-to-end workflow that prepares historical water-quality observations, estimates the **Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI)**, compares machine-learning approaches, explains predictions with **SHAP**, and turns detected parameter exceedances into monitoring recommendations.

| 🧹 Prepare | 🤖 Predict | 🔎 Explain | 🧭 Recommend |
|:--:|:--:|:--:|:--:|
| Inspect, clean and analyse missing data | Train and compare ML/DL models | Identify influential water-quality parameters | Surface possible pressures and follow-up actions |

> **Important:** This is a research and decision-support prototype. Predictions and rule-based recommendations are not a substitute for laboratory confirmation, local regulatory standards, or professional environmental assessment.

## ✨ What makes it interesting?

- **Data-quality first:** Separate scripts inspect raw data, validate values, analyse missingness, and build station-year datasets.
- **Baseline vs. imputed comparison:** Experiments compare modelling on baseline data with modelling on an imputed dataset.
- **Multiple model families:** Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost, DNN and TabNet experiments.
- **Explainable AI:** SHAP global importance, summary/dependence plots, and local explanations help interpret model behaviour.
- **Decision support:** A recommendation engine checks selected parameter thresholds and provides suggested follow-up monitoring actions.
- **Interactive dashboard:** A deployed Streamlit application lets users enter water-quality measurements and examine prediction results.

## 🧠 How the system works

```mermaid
flowchart TD
    A[Historical river water-quality data] --> B[Data inspection and quality checks]
    B --> C[Missingness and parameter analysis]
    C --> D[Preprocessing and station-year aggregation]
    D --> E[WQI calculation and validation]
    D --> F[Missing-value imputation and validation]
    E --> G[Baseline modelling dataset]
    F --> H[Imputed modelling dataset]
    G --> I[Train and compare ML / DL models]
    H --> I
    I --> J[Best-performing candidate evaluation]
    J --> K[SHAP model explanations]
    J --> L[Recommendation engine]
    L --> M[Streamlit decision-support dashboard]
    K --> M
