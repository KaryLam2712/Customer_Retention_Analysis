import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

telco_churn_data = pd.read_csv('/Users/karylam/Desktop/Customer Churn/Telco-Customer-Churn.csv')

# pie chart
l = list(telco_churn_data['Churn'].value_counts())
circle = [l[0] / sum(l) * 100, l[1] / sum(l) * 100]

labels = ['Not-Churn Customer', 'Churn Customer']
counts = telco_churn_data['Churn'].value_counts()

def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({val} customers)'
    return my_format

plt.figure(figsize=(10,6))
patches, texts, autotexts =plt.pie(circle, 
                                   labels=labels,
                                   autopct=autopct_format(counts),
                                   startangle=90,
                                   explode=(0.1, 0),
                                   colors=['#EEBA30', '#2A4D69'],
                                   wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True})
autotexts[1].set_color('white')
for autotext in autotexts:
    autotext.set_fontsize(14) 

for text in texts:
    text.set_fontsize(18)

plt.title('Churn vs Non-Churn', fontsize=22)
plt.axis('equal')
plt.show()


# comparison table 
telco_copy = telco_churn_data.copy()

telco_copy['TotalCharges'] = pd.to_numeric(telco_copy['TotalCharges'], errors='coerce')
telco_copy = telco_copy.dropna(subset=['TotalCharges'])
telco_copy.drop(columns=['customerID'], axis=1, inplace=True)

# Label encoding for categorical columns
le = LabelEncoder()
df1 = telco_copy.copy()
categorical_cols = df1.select_dtypes(include=['object']).columns  
for col in categorical_cols:
    df1[col] = le.fit_transform(df1[col])

columns_to_drop = ['Churn', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaperlessBilling', 'PaymentMethod']
df1_dropped = df1.drop(columns=columns_to_drop)

churn_dropped = df1_dropped[df1['Churn'] == 1].describe().T
not_churn_dropped = df1_dropped[df1['Churn'] == 0].describe().T

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

# Churned customers heatmap
plt.subplot(1, 2, 1)
sns.heatmap(churn_dropped[['mean']], annot=True, cmap=['#2A4D69'], 
            annot_kws={"fontsize": 12}, linewidths=0.3, linecolor='black', cbar=False, fmt='.2f')
plt.yticks(fontsize=10) 
plt.title('Churned Customers', fontsize=16)

# Not Churned customers heatmap
plt.subplot(1, 2, 2)
sns.heatmap(not_churn_dropped[['mean']], annot=True, cmap=['#EEBA30'], 
            annot_kws={"fontsize": 12}, linewidths=0.3, linecolor='black', cbar=False, fmt='.2f')
plt.yticks(fontsize=10) 
plt.title('Not Churned Customers', fontsize=16)

fig.tight_layout(pad=2)
plt.show()

categorical_vars = ["gender", "SeniorCitizen", "Dependents", "Partner"]

# Loop through each variable and create a stacked bar plot with churn percentage
for var in categorical_vars:
    # Calculate counts for each combination of the variable and churn status
    churn_counts = telco_churn_data.groupby([var, 'Churn']).size().unstack()

    # Convert counts to percentages
    churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100

    # Plot the stacked bar plot with percentages
    churn_percentages.plot(kind='bar', stacked=True, color=['#EEBA30', '#222F5B'], figsize=(8, 5))
    plt.title(f"Churn Percentage by {var}")
    plt.xlabel(var)
    plt.ylabel("Churn Percentage (%)")
    plt.legend(title="Churn", labels=["No", "Yes"])

    # Set x-axis labels to horizontal
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()




# plot graph
for i, predictor in enumerate(telco_churn_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i, figsize=(8, 5))
    sns.countplot(data=telco_churn_data, x=predictor, hue='Churn', palette={'No': '#222F5B', 'Yes': '#EEBA30'})
    plt.title(f'Churn by {predictor}')
    plt.tight_layout()
plt.show()

# angle 
telco_copy2 = telco_churn_data.copy()

telco_copy2['tenure_group'] = pd.cut(telco_copy2['tenure'], bins=[0, 24, 100], labels=['0-24 months', '>24 months'])

churn_rate_by_tenure = telco_copy2.groupby('tenure_group')['Churn'].value_counts(normalize=True).unstack().fillna(0)
print(churn_rate_by_tenure)
labels = ['0-24 months', '>24 months']
churn_no = churn_rate_by_tenure['No'].values
churn_yes = churn_rate_by_tenure['Yes'].values

fig, ax = plt.subplots()

bar_width = 0.35
index = range(len(labels))

colors_no_churn = ['#EEBA30', '#EEBA30']  
colors_yes_churn = ['#2A4D69', '#2A4D69']  

bar1 = ax.bar(index, churn_no, bar_width, label='Not Churn', color=colors_no_churn)
bar2 = ax.bar([i + bar_width for i in index], churn_yes, bar_width, label='Churn', color=colors_yes_churn)

for i in index:
    ax.text(i, churn_no[i] + 0.02, f'{churn_no[i]*100:.1f}%', ha='center', va='bottom', fontsize=12)
    ax.text(i + bar_width, churn_yes[i] + 0.02, f'{churn_yes[i]*100:.1f}%', ha='center', va='bottom', fontsize=12)

ax.set_xlabel('Tenure Group', fontsize=14)
ax.set_ylabel('Churn Rate', fontsize=14)
ax.set_title('Churn Rate vs Tenure Group',fontsize=16)
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(labels, fontsize=14)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()

telco_copy2.to_csv('telco_eda', index=False)  


