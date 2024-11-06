import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


telco_data = pd.read_csv("/Users/karylam/Desktop/Customer Churn/telco_data.csv")


### 2. Churn Rate Analysis
# Overall Churn Rate
churn_rate = telco_data['Churn'].value_counts(normalize=True) * 100
print(f"Overall Churn Rate:\n{churn_rate}\n")

# Churn by Tenure Category
churn_by_tenure = telco_data.groupby('tenure')['Churn'].value_counts(normalize=True).unstack() * 100
churn_by_tenure.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Tenure Category')
plt.ylabel('Percentage')
plt.show()

### 3. Demographic Analysis
# Churn by Gender and Senior Citizen Status
sns.countplot(data=telco_data, x='gender', hue='Churn')
plt.title('Churn by Gender')
plt.show()

sns.countplot(data=telco_data, x='SeniorCitizen', hue='Churn')
plt.title('Churn by Senior Citizen Status')
plt.show()

# Churn by Partner and Dependents
sns.countplot(data=telco_data, x='Partner', hue='Churn')
plt.title('Churn by Partner Status')
plt.show()

sns.countplot(data=telco_data, x='Dependents', hue='Churn')
plt.title('Churn by Dependents')
plt.show()

### 4. Churn by Contract Type (Tariff Plan Churn)
# Churn by Contract, Paperless Billing, and Payment Method
sns.countplot(data=telco_data, x='Contract', hue='Churn')
plt.title('Churn by Contract Type')
plt.show()

sns.countplot(data=telco_data, x='PaperlessBilling', hue='Churn')
plt.title('Churn by Paperless Billing')
plt.show()

sns.countplot(data=telco_data, x='PaymentMethod', hue='Churn')
plt.title('Churn by Payment Method')
plt.show()

### 5. Product Churn Analysis
# Churn by Phone Service, Internet Service, and other add-ons
sns.countplot(data=telco_data, x='PhoneService', hue='Churn')
plt.title('Churn by Phone Service')
plt.show()

sns.countplot(data=telco_data, x='InternetService', hue='Churn')
plt.title('Churn by Internet Service')
plt.show()

# Churn by StreamingTV, OnlineSecurity, DeviceProtection, etc.
addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for addon in addons:
    sns.countplot(data=telco_data, x=addon, hue='Churn')
    plt.title(f'Churn by {addon}')
    plt.show()

### 6. Usage Churn Analysis
# Churn by Tenure and Service Usage
sns.boxplot(data=telco_data, x='tenure', y='MonthlyCharges', hue='Churn')
plt.title('Monthly Charges by Tenure and Churn')
plt.show()

### 7. Monthly and Total Charges Analysis
# Churn by Monthly and Total Charges
sns.boxplot(data=telco_data, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn')
plt.show()

sns.boxplot(data=telco_data, x='Churn', y='TotalCharges')
plt.title('Total Charges by Churn')
plt.show()

### 8. Correlation Heatmap
# Convert categorical columns to numeric (if needed for correlation)
df_encoded = pd.get_dummies(telco_data, drop_first=True)

# Generate correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(telco_data['tenure'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Customer Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.show()

churn_rate_by_tenure = telco_data.groupby('tenure')['Churn'].value_counts(normalize=True).unstack()['Yes'] * 100
churn_rate_by_tenure.plot(kind='line', figsize=(10, 6), marker='o')
plt.title('Churn Rate by Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Churn Rate (%)')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(telco_data[telco_data['Churn'] == 'Yes']['MonthlyCharges'], label='Churned', fill=True)
sns.kdeplot(telco_data[telco_data['Churn'] == 'No']['MonthlyCharges'], label='Retained', fill=True)
plt.title('Monthly Charges Distribution by Churn Status')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.legend()
plt.show()


contract_churn = telco_data[telco_data['Churn'] == 'Yes'].groupby(['Contract', 'tenure']).size().unstack().fillna(0)
contract_churn.T.plot(figsize=(12, 6), marker='o')
plt.title('Churn by Contract Type Over Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Churned Customers')
plt.legend(title='Contract Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=telco_data, x='tenure', y='MonthlyCharges', hue='Churn')
plt.title('Monthly Charges by Tenure Category and Churn Status')
plt.xlabel('Tenure Category')
plt.ylabel('Monthly Charges')
plt.legend(title='Churn')
plt.show()

payment_churn = telco_data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack() * 100
payment_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Churn Rate by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Percentage')
plt.legend(title='Churn')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=telco_data, x='MonthlyCharges', y='TotalCharges', hue='Churn', alpha=0.7)
plt.title('Monthly Charges vs. Total Charges by Churn Status')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.legend(title='Churn')
plt.show()

add_ons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for service in add_ons:
    service_churn = telco_data.groupby(service)['Churn'].value_counts(normalize=True).unstack() * 100
    service_churn.plot(kind='bar', stacked=True, figsize=(8, 4), color=['skyblue', 'salmon'])
    plt.title(f'Churn Rate by {service}')
    plt.ylabel('Percentage')
    plt.legend(title='Churn')
    plt.show()

    # Monthly Charges Distribution
plt.figure(figsize=(10, 5))
sns.histplot(telco_data['MonthlyCharges'], kde=True)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.show()

# Total Charges Distribution
plt.figure(figsize=(10, 5))
sns.histplot(telco_data['TotalCharges'], kde=True)
plt.title('Total Charges Distribution')
plt.xlabel('Total Charges')
plt.ylabel('Density')
plt.show()

telco_data['LogMonthlyCharges'] = np.log(telco_data['MonthlyCharges'] + 1)  

# Plot Log-Transformed Monthly Charges
plt.figure(figsize=(10, 5))
sns.histplot(telco_data['LogMonthlyCharges'], kde=True)
plt.title('Log-Transformed Monthly Charges Distribution')
plt.xlabel('Log of Monthly Charges')
plt.ylabel('Density')
plt.show()
