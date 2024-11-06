import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as st

# Load the data
df = pd.read_csv('/Users/karylam/Desktop/Customer Churn/Telco-Customer-Churn.csv')

# Convert 'Churn' to binary (1 for "Yes", 0 for "No")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Define time and event variables
timevar = df['tenure']
eventvar = df['Churn']

# Create dummy variables for categorical features
categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']
survivaldata = pd.get_dummies(df, columns=categorical, drop_first=True)

# Initialize Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Plot overall survival curve
plt.figure()
kmf.fit(timevar, event_observed=eventvar, label="All Customers")
kmf.plot()
plt.ylabel('Probability of Customer Survival')
plt.xlabel('Tenure')
plt.title('Kaplan-Meier Curve')
plt.show()

# Survival analysis by specific groups (e.g., Senior Citizen)
SeniorCitizen = (survivaldata['SeniorCitizen_1'] == 1)
no_SeniorCitizen = (survivaldata['SeniorCitizen_1'] == 0)

plt.figure()
ax = plt.subplot(1,1,1)

# Kaplan-Meier fit for senior citizens
kmf.fit(timevar[SeniorCitizen], event_observed=eventvar[SeniorCitizen], label="Senior Citizen")
kmf.plot(ax=ax)

# Kaplan-Meier fit for non-senior citizens
kmf.fit(timevar[no_SeniorCitizen], event_observed=eventvar[no_SeniorCitizen], label="Non Senior Citizen")
kmf.plot(ax=ax)

plt.title('Survival of customers: Senior Citizen')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.show()

# Conduct log-rank test
groups = logrank_test(timevar[SeniorCitizen], timevar[no_SeniorCitizen], 
                      event_observed_A=eventvar[SeniorCitizen], event_observed_B=eventvar[no_SeniorCitizen])
groups.print_summary()


#  payment method 
# Define payment method groups with extra checks
automatic_Credit_Card = (survivaldata.get('PaymentMethod_Credit card (automatic)', 0) == 1)
electronic_check = (survivaldata.get('PaymentMethod_Electronic check', 0) == 1)
mailed_check = (survivaldata.get('PaymentMethod_Mailed check', 0) == 1)
automatic_Bank_Transfer = (~automatic_Credit_Card & ~electronic_check & ~mailed_check)

# Print counts of each payment method group for verification
print("Automatic Credit Card:", automatic_Credit_Card.sum())
print("Electronic Check:", electronic_check.sum())
print("Mailed Check:", mailed_check.sum())
print("Automatic Bank Transfer:", automatic_Bank_Transfer.sum())

# Plot Kaplan-Meier curves for each payment method
plt.figure()
ax = plt.subplot(1, 1, 1)

# Fit and plot each payment method group
kmf.fit(timevar[automatic_Credit_Card], event_observed=eventvar[automatic_Credit_Card], label="Automatic Credit Card")
kmf.plot(ax=ax)

kmf.fit(timevar[electronic_check], event_observed=eventvar[electronic_check], label="Electronic Check")
kmf.plot(ax=ax)

kmf.fit(timevar[mailed_check], event_observed=eventvar[mailed_check], label="Mailed Check")
kmf.plot(ax=ax)

kmf.fit(timevar[automatic_Bank_Transfer], event_observed=eventvar[automatic_Bank_Transfer], label="Automatic Bank Transfer")
kmf.plot(ax=ax)

plt.title('Survival of customers: Payment Method')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()

# Conduct Log-Rank Test for all payment methods
twoplusgroups_logrank = multivariate_logrank_test(df['tenure'], df['PaymentMethod'], df['Churn'], alpha=0.95)
twoplusgroups_logrank.print_summary()

# Movie
# One-hot encode categorical variables for survival analysis
categorical_vars = ['StreamingMovies', 'InternetService']
survivaldata = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Define the conditions for the groups based on StreamingMovies
try:
    no_internetService = survivaldata['StreamingMovies_No internet service'] == 1
    StreamingMovies = survivaldata['StreamingMovies_Yes'] == 1
    no_StreamingMovies = (survivaldata['StreamingMovies_No internet service'] == 0) & (survivaldata['StreamingMovies_Yes'] == 0)
except KeyError:
    print("Error: Required columns for StreamingMovies were not found in survivaldata. Check one-hot encoding.")
    raise

# Instantiate Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Plot the Kaplan-Meier curves
plt.figure(figsize=(8, 6))
ax = plt.subplot(1, 1, 1)

# Plot each group if data exists
if no_internetService.any():
    kmf.fit(timevar[no_internetService], event_observed=eventvar[no_internetService], label="No Internet Service")
    kmf.plot(ax=ax)

if StreamingMovies.any():
    kmf.fit(timevar[StreamingMovies], event_observed=eventvar[StreamingMovies], label="Streaming Movies")
    kmf.plot(ax=ax)

if no_StreamingMovies.any():
    kmf.fit(timevar[no_StreamingMovies], event_observed=eventvar[no_StreamingMovies], label="No Streaming Movies")
    kmf.plot(ax=ax)

# Add plot labels and title
plt.title('Survival of Customers: Streaming Movies')
plt.xlabel('Tenure')
plt.ylabel('Survival Probability')
plt.yticks(np.linspace(0, 1, 11))

# Display the plot
plt.tight_layout()
plt.show()

# Conduct the multivariate log-rank test
try:
    logrank_results = multivariate_logrank_test(df['tenure'], df['StreamingMovies'], df['Churn'], alpha=0.95)
    logrank_results.print_summary()
except KeyError:
    print("Error: Log-rank test variable 'StreamingMovies' not found in the original data. Check data preparation.")

    