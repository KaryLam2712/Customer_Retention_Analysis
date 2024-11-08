import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.stats as stats

telco_churn_data = pd.read_csv('/Users/karylam/Desktop/Customer Churn/Telco-Customer-Churn.csv')

telco_churn_data['Churn_binary'] = telco_churn_data['Churn'].map({'Yes': 1, 'No': 0})

# Create a cross-tabulation table for PaymentMethod and Churn
contingency_table = pd.crosstab(telco_churn_data['PaymentMethod'], telco_churn_data['Churn'])

# Perform chi-square test
chi2 = stats.chi2_contingency(contingency_table)[0]

# Calculate Cramér's V
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Cramér's V between Payment Method and Churn: {cramers_v:.2f}")

# stackabr internet service 
internet_service_var = "InternetService"

# Calculate counts for each combination of InternetService and Churn status
churn_counts = telco_churn_data.groupby([internet_service_var, 'Churn']).size().unstack()

# Convert counts to percentages
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100

# Plot the stacked bar plot with percentages
churn_percentages.plot(kind='bar', stacked=True, color=['#EEBA30', '#222F5B'], figsize=(8, 5))
plt.title("Churn Percentage by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])

# Set x-axis labels to horizontal for readability
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

plt.figure(figsize=(16,8))
sns.countplot(x="tenure", hue="Churn", data=telco_churn_data)
plt.show()


# stackbar chart for payment method 

payment_method_var = "PaymentMethod"

# Calculate counts for each combination of PaymentMethod and Churn status
churn_counts = telco_churn_data.groupby([payment_method_var, 'Churn']).size().unstack(fill_value=0)

# Convert counts to percentages
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100

# Plot the stacked bar plot with percentages
churn_percentages.plot(kind='bar', stacked=True, color=['#EEBA30', '#222F5B'], figsize=(8, 5))
plt.title("Churn Percentage by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])

# Set x-axis labels to horizontal for readability
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# barchart senior citizen vs paymeent 
churn_counts = telco_churn_data.groupby(['PaymentMethod', 'SeniorCitizen', 'Churn']).size().unstack(fill_value=0)

# Calculate churn percentage (only for 'Yes' churn)
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
churn_percentages = churn_percentages['Yes'].unstack()  # Focus on churn percentage for 'Yes'

# Reset index for plotting
churn_percentages = churn_percentages.reset_index()

# Melt the DataFrame for easier plotting with Seaborn
plot_data = churn_percentages.melt(id_vars=['PaymentMethod'], var_name='SeniorCitizen', value_name='Churn Percentage')

# Ensure SeniorCitizen column is categorical for hue mapping
plot_data['SeniorCitizen'] = plot_data['SeniorCitizen'].map({0: "No", 1: "Yes"})

# Plot the churn percentage as a grouped bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_data, x='PaymentMethod', y='Churn Percentage', hue='SeniorCitizen', 
            palette={'No': '#66cccc', 'Yes': '#d43d1a'})

# Set plot titles and labels
plt.title("Churn Percentage by Payment Method and Senior Citizen Status")
plt.xlabel("Payment Method")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Senior Citizen")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#  graph 
# Calculate counts for each combination of InternetService, Dependents, and Churn
churn_counts = telco_churn_data.groupby(['InternetService', 'Dependents', 'Churn']).size().unstack(fill_value=0)

# Calculate churn percentage (only for 'Yes' churn)
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
churn_percentages = churn_percentages['Yes'].unstack()  # Focus on churn percentage for 'Yes'

# Reset index for plotting
churn_percentages = churn_percentages.reset_index()

# Melt the DataFrame for easier plotting with Seaborn
plot_data = churn_percentages.melt(id_vars=['InternetService'], var_name='Dependents', value_name='Churn Percentage')

# Plot the churn percentage as a grouped bar chart
plt.figure(figsize=(8, 5))
sns.barplot(data=plot_data, x='InternetService', y='Churn Percentage', hue='Dependents', palette={'Yes': '#cff65e', 'No': '#ffa550'})

# Set plot titles and labels
plt.title("Churn Percentage by Internet Service and Dependents")
plt.xlabel("Internet Service")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Dependents")  # Use automatic labels generated by Seaborn for consistency

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()




# Plot tenure distribution by Online Security Service Subscription 

sns.kdeplot(telco_churn_data.tenure[telco_churn_data.OnlineSecurity == "No"], label="No", color="orange", linewidth=2)
sns.kdeplot(telco_churn_data.tenure[telco_churn_data.OnlineSecurity == "Yes"], label="Yes", color="blue", linewidth=2)
sns.kdeplot(telco_churn_data.tenure[telco_churn_data.OnlineSecurity == "No internet service"], label="No Internet Service", color="green", linewidth=2)
plt.title("Tenure Distribution by Online Security Service Subscription")
plt.legend()
plt.show()


sns.kdeplot(telco_churn_data.tenure[telco_churn_data.StreamingTV == "No"], label="No", color="orange", linewidth=2)
sns.kdeplot(telco_churn_data.tenure[telco_churn_data.StreamingTV == "Yes"], label="Yes", color="blue", linewidth=2)
sns.kdeplot(telco_churn_data.tenure[telco_churn_data.StreamingTV == "No internet service"], label="No Internet Service", color="green", linewidth=2)
plt.title("Tenure Distribution by Streaming TV Service Subscription")
plt.legend()
plt.show()


# sns.kdeplot(telco_churn_data.tenure[telco_churn_data.StreamingMovies == "No"], label="No", color="orange", linewidth=2)
# sns.kdeplot(telco_churn_data.tenure[telco_churn_data.StreamingMovies == "Yes"], label="Yes", color="blue", linewidth=2)
# sns.kdeplot(telco_churn_data.tenure[telco_churn_data.StreamingMovies == "No internet service"], label="No Internet Service", color="green", linewidth=2)
# plt.title("Tenure Distribution by Streaming Movies Service Subscription")
# plt.legend()
# plt.show()


sns.distplot(telco_churn_data.tenure[telco_churn_data.InternetService == "No"], hist_kws=dict(alpha=0.3), label="No")
sns.distplot(telco_churn_data.tenure[telco_churn_data.InternetService == "DSL"], hist_kws=dict(alpha=0.3), label="DSL")
sns.distplot(telco_churn_data.tenure[telco_churn_data.InternetService == "Fiber optic"], hist_kws=dict(alpha=0.3), label="Fiber optic")
plt.title("Tenure Distribution by Internet Service type")
plt.legend()
plt.show()


#  facet bar chart 
# Define factors to analyze
factors = ['TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity']

# Filter out rows with "No internet service" for relevant columns
telco_churn_data_filtered = telco_churn_data[
    ~(
        (telco_churn_data['TechSupport'] == 'No internet service') |
        (telco_churn_data['DeviceProtection'] == 'No internet service') |
        (telco_churn_data['OnlineBackup'] == 'No internet service') |
        (telco_churn_data['OnlineSecurity'] == 'No internet service') 
    )
]

# Initialize an empty list to collect processed data
processed_data = []

# Loop over each factor and calculate churn percentages
for factor in factors:
    churn_data = telco_churn_data_filtered.groupby([factor, 'Churn']).size().unstack(fill_value=0)
    churn_data = churn_data.div(churn_data.sum(axis=1), axis=0) * 100  # Convert to percentages
    churn_data = churn_data.reset_index()  # Reset index for plotting
    churn_data['Factor'] = factor  # Add column to indicate factor name
    churn_data = churn_data.rename(columns={factor: 'Category'})  # Rename for consistency
    processed_data.append(churn_data)  # Append to list

# Concatenate all factors into a single DataFrame
plot_data = pd.concat(processed_data, ignore_index=True)

# Melt the DataFrame for easier plotting with Seaborn
plot_data = plot_data.melt(id_vars=['Factor', 'Category'], 
                           var_name='Churn', value_name='Percentage')

# Plot with catplot
g = sns.catplot(
    data=plot_data, kind="bar",
    x="Category", y="Percentage", hue="Churn", 
    col="Factor", col_wrap=2, height=4, aspect=0.8,
    palette={'Yes': '#222F5B', 'No': '#EEBA30'}
)

# Add percentage labels on each bar
for ax in g.axes.flatten():
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type="edge", padding=2)

# Set plot labels and titles
g.set_axis_labels("Category", "Churn Percentage (%)")
g.set_titles("{col_name}")
plt.suptitle("Churn Percentage by Factor and Category (Excluding No Internet Service)", y=1.05)
plt.tight_layout()
plt.show()

#  facet bar chart
telco_churn_data_filtered = telco_churn_data[
    (telco_churn_data['tenure'] >= 1) & 
    (telco_churn_data['tenure'] <= 24) &
    ~(
        (telco_churn_data['TechSupport'] == 'No internet service') |
        (telco_churn_data['DeviceProtection'] == 'No internet service') |
        (telco_churn_data['OnlineBackup'] == 'No internet service') |
        (telco_churn_data['OnlineSecurity'] == 'No internet service')
    )
]

# Define factors to analyze
factors = ['TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity']

# Initialize an empty list to collect processed data
processed_data = []

# Loop over each factor and calculate churn percentages
for factor in factors:
    churn_data = telco_churn_data_filtered.groupby([factor, 'Churn']).size().unstack(fill_value=0)
    churn_data = churn_data.div(churn_data.sum(axis=1), axis=0) * 100  # Convert to percentages
    churn_data = churn_data.reset_index()  # Reset index for plotting
    churn_data['Factor'] = factor  # Add column to indicate factor name
    churn_data = churn_data.rename(columns={factor: 'Category'})  # Rename for consistency
    processed_data.append(churn_data)  # Append to list

# Concatenate all factors into a single DataFrame
plot_data = pd.concat(processed_data, ignore_index=True)

# Melt the DataFrame for easier plotting with Seaborn
plot_data = plot_data.melt(id_vars=['Factor', 'Category'], 
                           var_name='Churn', value_name='Percentage')

# Plot with catplot
g = sns.catplot(
    data=plot_data, kind="bar",
    x="Category", y="Percentage", hue="Churn", 
    col="Factor", col_wrap=2, height=4, aspect=0.8,
    palette={'Yes': '#222F5B', 'No': '#EEBA30'}
)

# Add percentage labels on each bar
for ax in g.axes.flatten():
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type="edge", padding=2)

# Set plot labels and titles
g.set_axis_labels("Category", "Churn Percentage (Tenure 1-24 Months)")
g.set_titles("{col_name}")
plt.suptitle("Churn Percentage by Factor and Category (Tenure 1-24 Months)", y=1.05)
plt.tight_layout()
plt.show()

# KDE plot for Monthly Charges by Churn 
plt.figure(figsize=(10, 6))
Mth = sns.kdeplot(telco_churn_data.MonthlyCharges[telco_churn_data["Churn"] == "No"],
                  color="#e95c29", fill=True, linewidth=2, label="No Churn")
Mth = sns.kdeplot(telco_churn_data.MonthlyCharges[telco_churn_data["Churn"] == "Yes"],
                  color="#14293a", fill=True, linewidth=2, label="Churn")

# Customize the plot appearance
Mth.set_ylabel('Density', fontsize=18)
Mth.set_xlabel('Monthly Charges', fontsize=18)
Mth.set_title('Monthly Charges by Churn Status', fontsize=20)
plt.legend(title="Churn Status", title_fontsize=14, fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()