import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

telco_churn_data = pd.read_csv("Dataset/Telco-Customer-Churn.csv")

print("Dataset columns:", telco_churn_data.columns.values)
print("Dataset types:\n", telco_churn_data.dtypes)
print("Dataset description:\n", telco_churn_data.describe())

# missing data
missing_value_percentage = (
    telco_churn_data.isnull().sum() / len(telco_churn_data)
) * 100
print("\nMissing Value Percentages:\n", missing_value_percentage)

# Data cleaning
telco_data = telco_churn_data.copy()

telco_data["TotalCharges"] = pd.to_numeric(telco_data["TotalCharges"], errors="coerce")

telco_data = telco_data.dropna(subset=["TotalCharges"])

labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
telco_data["tenure_group"] = pd.cut(
    telco_data["tenure"], bins=range(1, 80, 12), right=False, labels=labels
)

telco_data.drop(columns=["customerID"], axis=1, inplace=True)

telco_data.to_csv("telco_data_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'telco_data_cleaned.csv'")
