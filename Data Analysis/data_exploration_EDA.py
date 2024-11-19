import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

telco_data = pd.read_csv("Dataset/telco_data_cleaned.csv")

# Pie chart for churn distribution
l = list(telco_data["Churn"].value_counts())
circle = [l[0] / sum(l) * 100, l[1] / sum(l) * 100]
labels = ["Not-Churn Customer", "Churn Customer"]
counts = telco_data["Churn"].value_counts()


def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({val} customers)"

    return my_format


plt.figure(figsize=(10, 6))
patches, texts, autotexts = plt.pie(
    circle,
    labels=labels,
    autopct=autopct_format(counts),
    startangle=90,
    explode=(0.1, 0),
    colors=["#EEBA30", "#2A4D69"],
    wedgeprops={"edgecolor": "black", "linewidth": 1, "antialiased": True},
)
autotexts[1].set_color("white")
for autotext in autotexts:
    autotext.set_fontsize(14)
for text in texts:
    text.set_fontsize(18)

plt.title("Churn vs Non-Churn", fontsize=22)
plt.axis("equal")
plt.show()

le = LabelEncoder()
df_encoded = telco_data.copy()
categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Heatmap visualization for churn and non-churn statistics
columns_to_drop = [
    "Churn",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
df_reduced = df_encoded.drop(columns=columns_to_drop)
churn_stats = df_reduced[df_encoded["Churn"] == 1].describe().T
non_churn_stats = df_reduced[df_encoded["Churn"] == 0].describe().T

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
plt.subplot(1, 2, 1)
sns.heatmap(
    churn_stats[["mean"]],
    annot=True,
    cmap=["#2A4D69"],
    annot_kws={"fontsize": 12},
    cbar=False,
)
plt.title("Churned Customers")

plt.subplot(1, 2, 2)
sns.heatmap(
    non_churn_stats[["mean"]],
    annot=True,
    cmap=["#EEBA30"],
    annot_kws={"fontsize": 12},
    cbar=False,
)
plt.title("Not Churned Customers")
plt.tight_layout()
plt.show()

# Stacked bar plots for categorical variables
categorical_vars = ["gender", "SeniorCitizen", "Dependents", "Partner"]
for var in categorical_vars:
    churn_counts = telco_data.groupby([var, "Churn"]).size().unstack()
    churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
    churn_percentages.plot(
        kind="bar", stacked=True, color=["#EEBA30", "#222F5B"], figsize=(8, 5)
    )
    plt.title(f"Churn Percentage by {var}")
    plt.xlabel(var)
    plt.ylabel("Churn Percentage (%)")
    plt.tight_layout()
    plt.show()

# Count plots for predictors showing percentages
predictors = telco_data.drop(
    columns=["Churn", "TotalCharges", "MonthlyCharges"]
).columns

for i, predictor in enumerate(predictors):
    churn_percentages = (
        telco_data.groupby([predictor, "Churn"]).size().unstack(fill_value=0)
    )
    churn_percentages = (
        churn_percentages.div(churn_percentages.sum(axis=1), axis=0) * 100
    )

    churn_percentages = churn_percentages.reset_index().melt(
        id_vars=predictor, var_name="Churn", value_name="Percentage"
    )

    plt.figure(i, figsize=(8, 5))
    sns.barplot(
        data=churn_percentages,
        x=predictor,
        y="Percentage",
        hue="Churn",
        palette={"No": "#222F5B", "Yes": "#EEBA30"},
    )
    plt.title(f"Churn Percentage by {predictor}")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.show()
