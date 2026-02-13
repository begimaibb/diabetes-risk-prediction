import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.title("Predicting and Preventing Diabetes")
st.markdown("### Machine Learning Risk Assessment based on 2023 BRFSS Data")

st.info("Please enter your health information below to receive a risk assessment.")

col1, col2 = st.columns(2)

@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

age_map = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80 or older"
}

edu_map = {
    1: "Never attended school / kindergarten",
    2: "Elementary (Grades 1-8)",
    3: "Some High School (Grades 9-11)",
    4: "High School Graduate",
    5: "Some College / Technical School",
    6: "College Graduate"
}

income_map = {
    1: "Less than $10,000", 2: "$10k to <$15k", 3: "$15k to <$20k", 
    4: "$20k to <$25k", 5: "$25k to <$35k", 6: "$35k to <$50k", 
    7: "$50k to <$75k", 8: "$75k to <$100k", 9: "$100k to <$150k", 
    10: "$150k to <$200k", 11: "$200,000 or more"
}


with col1:
    st.subheader("Demographics & Lifestyle")
    Sex = st.radio("Sex", ["Female", "Male"])
    Age = st.select_slider("Age", options=list(age_map.keys()), format_func=lambda x: age_map[x])
    Education = st.select_slider("Education Level", options=list(edu_map.keys()), format_func=lambda x: edu_map[x])
    Income = st.select_slider("Annual Income", options=list(income_map.keys()), format_func=lambda x: income_map[x])
    
    st.subheader("Health History")
    HighBP = st.radio("High blood pressure?", ["No", "Yes"])
    HighChol = st.radio("High cholesterol?", ["No", "Yes"])
    CholCheck = st.radio("Cholesterol check (last 5 years)?", ["No", "Yes"])
    Stroke = st.radio("Have you ever had a stroke?", ["No", "Yes"])
    HeartDiseaseorAttack = st.radio("Heart disease or heart attack?", ["No", "Yes"])

with col2:
    st.subheader("Physical Indicators")
    BMI = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    GenHlth = st.slider("General health (1=Excellent, 5=Poor)", 1, 5, 2)
    MentHlth = st.slider("Poor mental health days (past 30 days)", 0, 30, 0)
    PhysHlth = st.slider("Poor physical health days (past 30 days)", 0, 30, 0)
    DiffWalk = st.radio("Difficulty walking or climbing stairs?", ["No", "Yes"])
    
    st.subheader("Daily Habits")
    Smoker = st.radio("Smoked at least 100 cigarettes in life?", ["No", "Yes"])
    PhysActivity = st.radio("Physical activity in past 30 days?", ["No", "Yes"])
    HvyAlcoholConsump = st.radio("Heavy alcohol consumption?", ["No", "Yes"])
    AnyHealthcare = st.radio("Do you have health insurance?", ["No", "Yes"])
    NoDocbcCost = st.radio("Could not see doctor due to the cost?", ["No", "Yes"])


if st.button('Predict My Risk'):
    binary_map = {"No": 0, "Yes": 1}
    sex_map = {"Female": 0, "Male": 1}
    
    input_data = [
        binary_map[HighBP], binary_map[HighChol], binary_map[CholCheck], BMI,
        binary_map[Smoker], binary_map[Stroke], binary_map[HeartDiseaseorAttack],
        binary_map[PhysActivity], binary_map[HvyAlcoholConsump], binary_map[AnyHealthcare], binary_map[NoDocbcCost],
        GenHlth, MentHlth, PhysHlth, binary_map[DiffWalk], sex_map[Sex], Age,
        Education, Income
    ]
    
    feature_names = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    features_df = pd.DataFrame([input_data], columns=feature_names)
    
    prob = model.predict_proba(features_df)[0][1]
    
    threshold = 0.3
    
    st.markdown("---")
    if prob > threshold:
        st.error(f"### High Risk Identified (Risk Score: {prob:.2%})")
        st.warning("**Recommendation:** Based on your indicators, you may be at risk for diabetes. We recommend consulting a healthcare professional for a formal A1C blood test.")
    else:
        st.success(f"### Low Risk Identified (Risk Score: {prob:.2%})")
        st.info("Maintain your healthy habits! Regular exercise and a balanced diet are key to long-term prevention.")

st.sidebar.markdown("---")
st.sidebar.write("Developed for Big Data Analytics Final Project")