# Predicting and Preventing Diabetes  
### A Machine Learning Analysis of BRFSS Data

**Author:** Begimai Bolotbekova
**Course:** 322390 Big Data Analytics
**Institution:** National Taipei University of Technology  
**Date:** January 2024  

---

## Project Overview

This project applies machine learning techniques to predict diabetes risk using data from the **Behavioral Risk Factor Surveillance System (BRFSS)** provided by the CDC. The final model is deployed as a **Streamlit web application** for real-time risk assessment.

---

## Dataset

- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Observations:** Large-scale population survey
- **Features:** 22 variables including:
  - Demographics (age, sex, income, education)
  - Health behaviors (smoking, physical activity, diet, alcohol)
  - Health indicators (BMI, blood pressure, cholesterol, chronic conditions)

The dataset is **imbalanced**, with fewer diabetic cases than non-diabetic cases.

---

## Machine Learning Approach

- **Problem Type:** Binary classification
- **Model:** XGBoost Classifier
- **Target Variable:** Diabetes (0 = No, 1 = Yes)
- **Preprocessing:**
  - Merged diabetes type 1 and 2 into a single positive class
  - Feature selection and transformation
- **Optimization:** Hyperparameter tuning using **Optuna**


---

## Web Application

A **Streamlit-based web app** allows users to:
- Input health and lifestyle information
- Receive an immediate diabetes risk prediction
- Increase awareness and encourage preventive action
