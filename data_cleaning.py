import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import numpy as np

telco_churn_data = pd.read_csv('/Users/karylam/Desktop/Customer Churn/Telco-Customer-Churn.csv')

print(telco_churn_data.columns.values)
print(telco_churn_data.dtypes)
print(telco_churn_data.describe())

# missing data
missing_value_percentage = (telco_churn_data.isnull().sum() / len(telco_churn_data * 100))

missing = pd.DataFrame({
    'column': telco_churn_data.columns,
    'missing_percentage': missing_value_percentage
}).reset_index(drop=True)

plt.figure(figsize=(16,5))
axis = sns.pointplot(x='column', y='missing_percentage', data=missing, color='#2A4D69')
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()

# data cleaning 
telco_data = telco_churn_data.copy()

telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')

# print(telco_data.isnull().sum())
# print(telco_data.loc[telco_data['TotalCharges'].isnull()])

telco_data = telco_data.dropna(subset=['TotalCharges'])
# print(telco_data.isnull().sum())

# Group the tenure in bins of 12 months
# print(telco_data['tenure'].max()) #72

# create bins 
labels = ["{0} - {1}".format(i , i + 11) for i in range (1,72,12)]
telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1,80,12), right=False, labels=labels)
# print(telco_data['tenure_group'].value_counts())

telco_data.drop(columns= ['customerID'], axis=1, inplace=True)
# print(telco_data.head())

telco_data.to_csv('telco_data.csv', index=False)  
