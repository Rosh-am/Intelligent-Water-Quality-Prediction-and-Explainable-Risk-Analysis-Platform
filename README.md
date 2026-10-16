<div align="center">

# 💧 Water Quality Risk Assessment

### Explainable Machine Learning for Water Quality Prediction & Decision Support

**From historical river measurements to interpretable water-quality estimates and actionable monitoring guidance.**

<p>
  <a href="https://water-quality-risk-assessment.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Launch_App-16A34A?style=for-the-badge" alt="Launch Live Demo">
  </a>
  <a href="https://github.com/Rosh-am/Intelligent-Water-Quality-Prediction-and-Explainable-Risk-Analysis-Platform">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github" alt="Source Code">
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Project-Capstone-0EA5E9?style=for-the-badge" alt="Capstone Project">
  <img src="https://img.shields.io/badge/Language-Python_3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ML-CatBoost_%7C_XGBoost-16A34A?style=for-the-badge" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Explainability-SHAP-7C3AED?style=for-the-badge" alt="Explainable AI">
  <img src="https://img.shields.io/badge/Deployment-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Deployment">
</p>

**An end-to-end AI/ML capstone project** combining water-quality data preparation, missing-value analysis, Water Quality Index modelling, machine-learning experiments, SHAP-based explainability, and an interactive decision-support dashboard.

### 🌐 Explore the Live Application

# [💧 Launch Water Quality Decision Support System →](https://water-quality-risk-assessment.streamlit.app/)

*Enter water-quality measurements, generate predictions, examine model explanations, and view decision-support recommendations.*

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Live Application](#-live-application)
- [System Architecture](#-system-architecture)
- [Water Quality Parameters](#-water-quality-parameters)
- [Machine Learning Models](#-machine-learning-models)
- [Model Benchmark Results](#-model-benchmark-results)
- [Explainable AI with SHAP](#-explainable-ai-with-shap)
- [Decision-Support Recommendation Engine](#-decision-support-recommendation-engine)
- [Dashboard Features](#-dashboard-features)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Running the Dashboard](#-running-the-dashboard)
- [Deployment](#-deployment)
- [Testing and Validation](#-testing-and-validation)
- [Responsible Use and Limitations](#-responsible-use-and-limitations)
- [Future Improvements](#-future-improvements)
- [Project Information](#-project-information)

---

## 🌊 Project Overview

Water quality is an important factor in environmental sustainability, public health, and the management of freshwater resources.

Historical water-quality datasets can contain missing observations, inconsistent measurements, and complex relationships between chemical parameters. Interpreting these measurements and converting them into actionable information can be challenging.

**Water Quality Risk Assessment** is an AI/ML-based decision-support project designed to analyse water-quality measurements, predict the Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI), explain model predictions, and provide preliminary monitoring recommendations.

The project combines data preprocessing, machine-learning model comparisons, missing-value handling, explainable artificial intelligence, and a web-based dashboard.

The dashboard is deployed publicly through Streamlit Community Cloud.

### 🚀 Quick Links

| Resource | Link |
|---|---|
| 🌐 Live Application | [Open Water Quality Decision Support System](https://water-quality-risk-assessment.streamlit.app/) |
| 💻 GitHub Repository | [View Source Code](https://github.com/Rosh-am/Intelligent-Water-Quality-Prediction-and-Explainable-Risk-Analysis-Platform) |
| 📊 Model Experiments | [`03_XGBoost`](03_XGBoost/) |
| 🔍 SHAP Analysis | [`04_SHAP`](04_SHAP/) |
| 🧭 Recommendation Engine | [`05_Recommendation_Engine`](05_Recommendation_Engine/) |
| 🖥️ Dashboard Source | [`06_Dashboard`](06_Dashboard/) |
| 📄 Research Paper | [`07_Research_Paper`](07_Research_Paper/) |
| 📘 Project Report | [`08_Project_Report`](08_Project_Report/) |
| 🎤 Presentation | [`09_Presentation`](09_Presentation/) |

---

## 🎯 Problem Statement

Water-quality assessment involves interpreting multiple physical and chemical measurements whose relationships may be nonlinear and difficult to evaluate manually.

Traditional data analysis can be limited when datasets contain missing values, numerous observations, or complex relationships between water-quality indicators.

Furthermore, a prediction alone does not explain which parameters influenced the model's output or what measurements might deserve further investigation.

This project addresses these challenges through a workflow that:

1. Inspects and prepares historical water-quality data.
2. Analyses missing values and data-quality issues.
3. Calculates or processes Water Quality Index targets.
4. Trains and compares machine-learning models.
5. Explains model predictions using SHAP.
6. Checks selected parameter thresholds.
7. Presents predictions and recommendations through an interactive dashboard.

The goal is to make water-quality modelling more interpretable and accessible for preliminary analysis and decision support.

## 🎯 Project Objectives

- Develop a structured workflow for water-quality data preparation.
- Analyse missing observations and evaluate imputation strategies.
- Build machine-learning models for CCME WQI prediction.
- Compare model performance using MAE, RMSE, and R².
- Investigate the influence of individual water-quality parameters.
- Apply SHAP to interpret model predictions.
- Develop a rule-based recommendation engine for preliminary monitoring guidance.
- Integrate the modelling workflow into an interactive Streamlit application.
- Deploy the application publicly and maintain the source code on GitHub.

---

## ✨ Key Features

### 🧹 1. Data Quality Assessment

- Inspect historical water-quality records.
- Examine missing observations.
- Validate selected measurements.
- Prepare datasets for modelling.
- Analyse station-year observations where applicable.

### 🤖 2. Machine Learning

- Train and compare multiple regression models.
- Experiment with baseline and imputed datasets.
- Evaluate predictions using standard regression metrics.
- Save model outputs and comparison results.

### 🔍 3. Explainable Artificial Intelligence

- Calculate SHAP feature contributions.
- Examine influential water-quality parameters.
- Present local prediction explanations.
- Help users understand how individual features affect a model's output.

### 🧭 4. Decision-Support Recommendations

- Check selected water-quality parameters against configured thresholds.
- Display a recommendation priority.
- Report detected guideline exceedances according to the implemented rules.
- Provide preliminary follow-up monitoring guidance.

### 🖥️ 5. Interactive Dashboard

- Enter water-quality measurements.
- Select an available modelling dataset.
- Generate predictions.
- View the predicted CCME WQI and category.
- Inspect SHAP contributions.
- Review decision-support recommendations.

### ☁️ 6. Public Deployment

- Hosted using Streamlit Community Cloud.
- Accessible through a public HTTPS URL.
- Source code maintained in GitHub.
- Can be accessed without cloning the repository or installing Python locally.

---

## 🌐 Live Application

### [💧 Open the Deployed Dashboard](https://water-quality-risk-assessment.streamlit.app/)

**Application:** Water Quality Decision Support System

**Deployment platform:** Streamlit Community Cloud

**Source repository:** [Intelligent Water Quality Prediction and Explainable Risk Analysis Platform](https://github.com/Rosh-am/Intelligent-Water-Quality-Prediction-and-Explainable-Risk-Analysis-Platform)

### What can you do in the application?

1. Open the live application.
2. Select an available modelling dataset from the sidebar.
3. Enter the required water-quality measurements.
4. Click **Analyze Water Quality**.
5. Review the predicted CCME WQI and water-quality category.
6. Examine the SHAP feature contributions.
7. Review the decision-support priority and recommendation message.

### Current deployment status

The application has been launched successfully on Streamlit Community Cloud.

The deployed interface has displayed:

- Water-quality input controls.
- A model-dataset selection menu.
- A prediction results section.
- A SHAP contribution chart and corresponding table.
- A decision-support recommendation section.
- A recommendation priority indicator.
- A guideline-exceedance status message.

The live interface has been observed generating results. However, the successful deployment of the interface does not, by itself, establish that every prediction or recommendation is scientifically validated. Further testing against known cases is recommended.

---

## 🧠 System Architecture

The project follows an end-to-end data science workflow.

```mermaid
flowchart TD
    A[Historical Water Quality Data]
    B[Data Inspection and Quality Checks]
    C[Missingness and Parameter Analysis]
    D[Preprocessing and Dataset Preparation]
    E[WQI Calculation and Validation]
    F[Missing Value Imputation]
    G[Baseline Dataset]
    H[Imputed Dataset]
    I[Machine Learning Experiments]
    J[Model Evaluation and Comparison]
    K[Selected Trained Model]
    L[SHAP Explainability]
    M[Recommendation Engine]
    N[Streamlit Dashboard]
    O[Prediction and Decision Support]

    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    E --> G
    F --> H
    G --> I
    H --> I
    I --> J
    J --> K
    K --> L
    K --> M
    L --> N
    M --> N
    N --> O
