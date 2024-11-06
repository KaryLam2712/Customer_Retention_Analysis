import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick 
import matplotlib.pyplot as plt 

telco_data = pd.read_csv("/Users/karylam/Desktop/Customer Churn/telco_data.csv")
print(telco_data.head())

churn_rate_by_tenure1 = telco_data.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack().fillna(0)
print(churn_rate_by_tenure1)

# Data for visualization
labels1 = ['0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61-72 months']

# Ensure that churn_no1 and churn_yes1 match the length of labels1
churn_no1 = churn_rate_by_tenure1['No'].values
churn_yes1 = churn_rate_by_tenure1['Yes'].values

fig, ax = plt.subplots()

bar_width1 = 0.35
index1 = range(len(labels1))

colors_no_churn1 = ['#EEBA30'] * len(labels1)  
colors_yes_churn1 = ['#222F5B'] * len(labels1) 

# Plot bars for "No Churn" and "Yes Churn"
bar3 = ax.bar(index1, churn_no1, bar_width1, label='Not Churn', color=colors_no_churn1)
bar4 = ax.bar([i + bar_width1 for i in index1], churn_yes1, bar_width1, label='Churn', color=colors_yes_churn1)

# Add percentage labels on top of the bars
for i in index1:
    ax.text(i, churn_no1[i] + 0.02, f'{churn_no1[i]*100:.1f}%', ha='center', va='bottom', fontsize=13)
    ax.text(i + bar_width1, churn_yes1[i] + 0.02, f'{churn_yes1[i]*100:.1f}%', ha='center', va='bottom', fontsize=13)

ax.set_xlabel('Tenure Group', fontsize=13)
ax.set_ylabel('Churn Rate', fontsize=13)
ax.set_title('Churn Rate vs Tenure Group', fontsize=16)
ax.set_xticks([i + bar_width1 / 2 for i in index1])
ax.set_xticklabels(labels1, fontsize=13)
ax.legend(fontsize=13)

plt.tight_layout()
# plt.show()


# # # plot grpah 
telco_data_filtered = telco_data[(telco_data['tenure'] >= 1) & (telco_data['tenure'] <= 24)]
# # Plot graphs based on the filtered data
# for i, predictor in enumerate(telco_data_filtered.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
#     plt.figure(i, figsize=(8, 5))
#     sns.countplot(data=telco_data_filtered, x=predictor, hue='Churn', palette={'No': '#222F5B', 'Yes': '#EEBA30'})
#     plt.title(f'Churn by {predictor}')
#     plt.tight_layout()
# # plt.show()


# Relationship between monthly charge and churn (Filtered data)
telco_data_filtered['Churn'] = np.where(telco_data_filtered.Churn == 'Yes', 1, 0)
telco_data_filtered_dummies = pd.get_dummies(telco_data_filtered)
print(telco_data_filtered_dummies.head())

plt.rcParams.update({'font.size': 16})




# KDE plot for Total Charges by Churn (Filtered data)
Tot = sns.kdeplot(telco_data_filtered_dummies.TotalCharges[(telco_data_filtered_dummies["Churn"] == 0)],
                  color="#e95c29", fill=True)
Tot = sns.kdeplot(telco_data_filtered_dummies.TotalCharges[(telco_data_filtered_dummies["Churn"] == 1)],
                  ax=Tot, color="#14293a", fill=True)
Tot.legend(["No Churn", "Churn"], loc='upper right', fontsize=16)
Tot.set_ylabel('Density',fontsize=18)
Tot.set_xlabel('Total Charges',fontsize=18)
Tot.set_title('Total charges by churn (1-24 months)', fontsize=20)
# plt.show()

# Correlation with churn (Filtered data)
correlation_data = telco_data_filtered_dummies.corr()['Churn'].sort_values(ascending=False)
norm = plt.Normalize(correlation_data.min(), correlation_data.max())
colors = plt.cm.twilight(norm(correlation_data.values))
plt.figure(figsize=(20, 8))
correlation_data.plot(kind='bar', color=colors)
# plt.show()


# heatmap
telco_data_filtered = telco_data.drop(columns=['customerID'], errors='ignore')

# Ensure 'Churn' is numeric (convert 'Yes'/'No' to 1/0)
if 'Churn' in telco_data_filtered.columns and telco_data_filtered['Churn'].dtype == 'object':
    telco_data_filtered['Churn'] = np.where(telco_data_filtered['Churn'] == 'Yes', 1, 0)

# Convert categorical columns into dummy/indicator variables, excluding 'Churn'
telco_data_filtered_dummies = pd.get_dummies(telco_data_filtered, drop_first=True)

# Ensure 'Churn' is in the dataframe after dummy conversion
if 'Churn' not in telco_data_filtered_dummies.columns:
    telco_data_filtered_dummies['Churn'] = telco_data_filtered['Churn']

# Define keywords to capture relevant columns, excluding 'tenure_group'
keywords = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Select columns dynamically based on keywords
selected_columns = [col for col in telco_data_filtered_dummies.columns if any(keyword in col for keyword in keywords)]
print("Selected columns for analysis:", selected_columns)

# Filter the dataset to include only the selected columns
telco_data_filtered = telco_data_filtered_dummies[selected_columns]

# Convert 'TotalCharges' to numeric, handling errors
telco_data_filtered['TotalCharges'] = pd.to_numeric(telco_data_filtered['TotalCharges'], errors='coerce')

# Calculate correlation matrix on the filtered data
correlation_matrix = telco_data_filtered.corr()

# Apply a threshold to show only high correlations (above 0.3 or below -0.3)
threshold = 0.3
filtered_correlation = correlation_matrix[(correlation_matrix >= threshold) | (correlation_matrix <= -threshold)]

# Create a mask for the upper triangle to avoid duplicate values
mask = np.triu(np.ones_like(filtered_correlation, dtype=bool))

# Plot the heatmap with smaller font for annotations and axis labels
plt.figure(figsize=(16, 12))
sns.heatmap(
    filtered_correlation, 
    mask=mask, 
    cmap='twilight', 
    center=0, 
    annot=True, 
    fmt=".2f", 
    annot_kws={"size": 8},  # Set font size for annotations
    linewidths=.5, 
    cbar_kws={'label': 'Correlation Coefficient'}
)

# Set smaller font size for axis labels
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.title("Heatmap of Correlations with Threshold |correlation| > 0.3")
plt.show()

# Select the top 5 features most correlated with Churn for reference
top_features = correlation_matrix['Churn'].abs().sort_values(ascending=False).index[:5].tolist()
if 'Churn' not in top_features:
    top_features.append('Churn')  # Ensure 'Churn' is included in the list

print("Top correlated features with Churn:", top_features)

# Group data by 'SeniorCitizen' and 'Churn' for additional analysis
grouped_data = telco_data_filtered.groupby(['SeniorCitizen', 'Churn']).agg({
    'MonthlyCharges': 'mean',
    'TotalCharges': 'mean',
    'tenure': 'mean'
}).reset_index()

# Display the grouped data for reference
print(grouped_data)


# Calculate correlation matrix(pairplot)
telco_data_filtered = telco_data.drop(columns=['customerID'], errors='ignore')

# Ensure 'Churn' is numeric (convert 'Yes'/'No' to 1/0)
if 'Churn' in telco_data_filtered.columns and telco_data_filtered['Churn'].dtype == 'object':
    telco_data_filtered['Churn'] = np.where(telco_data_filtered['Churn'] == 'Yes', 1, 0)

# Create 'tenure_group' based on 'tenure'
tenure_bins = [0, 12, 24, 36, 48, 60, 72]
tenure_labels = ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72']
telco_data_filtered['tenure_group'] = pd.cut(telco_data_filtered['tenure'], bins=tenure_bins, labels=tenure_labels, right=True)

# Convert categorical columns into dummy/indicator variables, excluding 'Churn'
telco_data_filtered_dummies = pd.get_dummies(telco_data_filtered, drop_first=True)

# Ensure 'Churn' is in the dataframe after dummy conversion
if 'Churn' not in telco_data_filtered_dummies.columns:
    telco_data_filtered_dummies['Churn'] = telco_data_filtered['Churn']

# Define keywords to capture relevant columns, including 'tenure_group'
keywords = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
]

# Select columns dynamically based on keywords
selected_columns = [col for col in telco_data_filtered_dummies.columns if any(keyword in col for keyword in keywords)]
print("Selected columns for analysis:", selected_columns)

# Filter the dataset to include only the selected columns
telco_data_filtered = telco_data_filtered_dummies[selected_columns]

# Convert 'TotalCharges' to numeric, handling errors
telco_data_filtered['TotalCharges'] = pd.to_numeric(telco_data_filtered['TotalCharges'], errors='coerce')

# Calculate correlation matrix on the filtered data
correlation_matrix = telco_data_filtered.corr()

# Select the top 5 features most correlated with Churn
top_features = correlation_matrix['Churn'].abs().sort_values(ascending=False).index[:5].tolist()
if 'Churn' not in top_features:
    top_features.append('Churn')  # Ensure 'Churn' is included in the list

print("Top correlated features with Churn:", top_features)

# Set up the figure size for better readability
plt.figure(figsize=(12, 10))

plt.rcParams.update({
    'axes.titlesize': 10,     # Title font size
    'axes.labelsize': 8,      # Axis label font size
    'xtick.labelsize': 7,     # X-axis tick font size
    'ytick.labelsize': 7,     # Y-axis tick font size
    'legend.fontsize': 8,     # Legend font size
    'font.size': 8            # General font size
})

# Plot the pairplot for the top correlated features with Churn
sns.pairplot(
    telco_data_filtered[top_features],
    hue="Churn",
    palette={0: "#2A4D69", 1: "#EEBA30"},
    plot_kws={'alpha': 0.6, 's': 20},
    diag_kind="kde",
    height=2.5,
    aspect=1
)

plt.suptitle("Pairplot of Top Features Correlated with Churn", y=1.05, fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Group data by 'tenure_group', 'SeniorCitizen', and 'Churn' for additional analysis
grouped_data = telco_data_filtered.groupby(['tenure_group', 'SeniorCitizen', 'Churn']).agg({
    'MonthlyCharges': 'mean',
    'TotalCharges': 'mean',
    'tenure': 'mean'
}).reset_index()

# Display the grouped data for reference
print(grouped_data)


# graph
telco_data_filtered['TotalCharges'] = pd.to_numeric(telco_data_filtered['TotalCharges'], errors='coerce')

# senior citizen vs charge  vs tenure 
grouped_data = telco_data_filtered.groupby(['tenure_group', 'SeniorCitizen', 'Churn']).agg({
    'MonthlyCharges': 'mean',
    'TotalCharges': 'mean'
}).reset_index()

# Step 5: Plot Monthly Charges with SeniorCitizen distinction
plt.figure(figsize=(14, 8))
sns.barplot(x='tenure_group', y='MonthlyCharges', hue='SeniorCitizen', data=grouped_data, palette={0: '#EEBA30', 1: '#2A4D69'})
plt.title('Average Monthly Charges by Tenure Group and Senior Citizen Status')
plt.xlabel('Tenure Group')
plt.ylabel('Average Monthly Charges')
plt.xticks(rotation=45)
plt.legend(title='Senior Citizen')
plt.tight_layout()
plt.show()

# Step 6: Plot Total Charges with SeniorCitizen distinction
plt.figure(figsize=(14, 8))
sns.barplot(x='tenure_group', y='TotalCharges', hue='SeniorCitizen', data=grouped_data, palette={0: '#EEBA30', 1: '#2A4D69'})
plt.title('Average Total Charges by Tenure Group and Senior Citizen Status')
plt.xlabel('Tenure Group')
plt.ylabel('Average Total Charges')
plt.xticks(rotation=45)
plt.legend(title='Senior Citizen')
plt.tight_layout()
plt.show()

# KDE plot for Monthly Charges by Senior Citizen (Filtered data)
Mth = sns.kdeplot(telco_data_filtered.MonthlyCharges[(telco_data_filtered["SeniorCitizen"] == 0)],
                  color="#EEBA30", fill=True)
Mth = sns.kdeplot(telco_data_filtered.MonthlyCharges[(telco_data_filtered["SeniorCitizen"] == 1)],
                  ax=Mth, color="#2A4D69", fill=True)
Mth.legend(["Not Senior Citizen", "Senior Citizen"], loc='upper right', fontsize=16)
Mth.set_ylabel('Density', fontsize=18)
Mth.set_xlabel('Monthly Charges', fontsize=18)
Mth.set_title('Monthly Charges by Senior Citizen Status', fontsize=20)
plt.show()

# KDE plot for Total Charges by Senior Citizen (Filtered data)
Tot = sns.kdeplot(telco_data_filtered.TotalCharges[(telco_data_filtered["SeniorCitizen"] == 0)],
                  color="#EEBA30", fill=True)
Tot = sns.kdeplot(telco_data_filtered.TotalCharges[(telco_data_filtered["SeniorCitizen"] == 1)],
                  ax=Tot, color="#2A4D69", fill=True)
Tot.legend(["Not Senior Citizen", "Senior Citizen"], loc='upper right', fontsize=16)
Tot.set_ylabel('Density', fontsize=18)
Tot.set_xlabel('Total Charges', fontsize=18)
Tot.set_title('Total Charges by Senior Citizen Status', fontsize=20)
plt.show()


