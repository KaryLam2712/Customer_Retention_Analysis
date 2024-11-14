import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import statsmodels.api as st
from sklearn.preprocessing import LabelEncoder
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test   
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

# Load the data
telco_data = pd.read_csv("/Users/karylam/Desktop/Customer Churn/telco_data.csv")

# Convert "Yes"/"No" in 'Churn' column to 1/0 for survival analysis
telco_data['Churn'] = telco_data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Define the event (Churn) and duration (tenure) variables
eventvar = telco_data['Churn']
timevar = telco_data['tenure']

# List of categorical variables to create dummy variables
categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']

# Create dummy variables for categorical columns, excluding the first level to avoid multicollinearity
survivaldata = pd.get_dummies(telco_data, columns=categorical, drop_first=True)

# Drop columns if they exist in survivaldata
columns_to_drop = ['customerID', 'tenure', 'Churn']
for col in columns_to_drop:
    if col in survivaldata.columns:
        survivaldata.drop(col, axis=1, inplace=True)

# Add a constant to the dataset for the Cox model
survivaldata = st.add_constant(survivaldata, prepend=False)

# Kaplan-Meier Fitter
kmf = KaplanMeierFitter()

# Fit the model on the duration and event variables
kmf.fit(timevar, event_observed=eventvar, label="Clients")

# Plot the Kaplan-Meier survival curve
kmf.plot(color="#222F5B")
plt.ylabel('Probability of Customer Survival')
plt.xlabel('Tenure')
plt.title('Kaplan-Meier Curve')
plt.xlim(left=0)
plt.show()
