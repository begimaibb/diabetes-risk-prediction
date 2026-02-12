import pandas as pd
import numpy as np

file_path = './data/LLCP2023.XPT' 
df = pd.read_sas(file_path, format='xport')

df['Diabetes_binary'] = df['DIABETE4'].replace({2: 0, 3: 0, 4: 0})

binary_map = {2: 0, 1: 1}

df['HighBP'] = df['_RFHYPE6'].replace({1: 0, 2: 1}) # _RFHYPE6: 1=No, 2=Yes
df['HighChol'] = df['_RFCHOL3'].replace({1: 0, 2: 1}) # _RFCHOL3: 1=No, 2=Yes
df['CholCheck'] = df['_CHOLCH3'].replace({2: 0}) 
df['Smoker'] = df['SMOKE100'].replace(binary_map)
df['Stroke'] = df['CVDSTRK3'].replace(binary_map)
df['HeartDiseaseorAttack'] = df['_MICHD'].replace({2: 0}) # _MICHD: 1=Yes, 2=No
df['PhysActivity'] = df['_TOTINDA'].replace(binary_map)
df['HvyAlcoholConsump'] = df['_RFDRHV8'].replace({1: 0, 2: 1}) # _RFDRHV8: 1=No, 2=Yes
df['AnyHealthcare'] = df['_HLTHPL1'].replace(binary_map)
df['NoDocbcCost'] = df['MEDCOST1'].replace(binary_map)
df['DiffWalk'] = df['DIFFWALK'].replace(binary_map)
df['Sex'] = df['_SEX'].replace({2: 0}) # 1=Male, 0=Female

if '_FRTLT1A' in df.columns:
    df['Fruits'] = df['_FRTLT1A'].replace({2: 0})
if '_VEGLT1A' in df.columns:
    df['Veggies'] = df['_VEGLT1A'].replace({2: 0})

df['BMI'] = df['_BMI5'] / 100
df['MentHlth'] = df['MENTHLTH'].replace({88: 0})
df['PhysHlth'] = df['PHYSHLTH'].replace({88: 0})


for col in ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']:
    df = df[df[col].isin([0, 1])]


final_features = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

df_final = df.rename(columns={'_AGEG5YR': 'Age', 'GENHLTH': 'GenHlth', 'EDUCA': 'Education', 'INCOME3': 'Income'})
df_final = df_final[final_features].dropna()

df_final.to_csv('./data/brfss_health_ind_2023_cleaned.csv', index=False)