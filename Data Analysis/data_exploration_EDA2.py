import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

telco_data = pd.read_csv("Dataset/telco_data_cleaned.csv")

# churn rate vs tenure group
churn_rate_by_tenure = (
    telco_data.groupby("tenure_group")["Churn"].value_counts(normalize=True).unstack()
)

labels1 = [
    "0-12 months",
    "13-24 months",
    "25-36 months",
    "37-48 months",
    "49-60 months",
    "61-72 months",
]

churn_no1 = churn_rate_by_tenure["No"].values
churn_yes1 = churn_rate_by_tenure["Yes"].values

fig, ax = plt.subplots(figsize=(10, 6))

bar_width1 = 0.35
index1 = range(len(labels1))

colors_no_churn1 = ["#EEBA30"] * len(labels1)
colors_yes_churn1 = ["#222F5B"] * len(labels1)

bar3 = ax.bar(index1, churn_no1, bar_width1, label="Not Churn", color=colors_no_churn1)
bar4 = ax.bar(
    [i + bar_width1 for i in index1],
    churn_yes1,
    bar_width1,
    label="Churn",
    color=colors_yes_churn1,
)

for i in index1:
    ax.text(
        i,
        churn_no1[i] + 0.02,
        f"{churn_no1[i]*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    ax.text(
        i + bar_width1,
        churn_yes1[i] + 0.02,
        f"{churn_yes1[i]*100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax.set_xlabel("Tenure Group", fontsize=12)
ax.set_ylabel("Churn Rate", fontsize=12)
ax.set_title("Churn Rate vs Tenure Group", fontsize=16)
ax.set_xticks([i + bar_width1 / 2 for i in index1])
ax.set_xticklabels(labels1, fontsize=11)
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()

# Add a binary churn column for statistical analysis
telco_data["Churn_binary"] = telco_data["Churn"].map({"Yes": 1, "No": 0})

# Chi-square test and Cramér's V for PaymentMethod and Churn
contingency_table = pd.crosstab(telco_data["PaymentMethod"], telco_data["Churn"])
chi2 = stats.chi2_contingency(contingency_table)[0]
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
print(f"Cramér's V between Payment Method and Churn: {cramers_v:.2f}")

# Stacked bar chart for Internet Service and Churn
internet_service_var = "InternetService"
churn_counts = telco_data.groupby([internet_service_var, "Churn"]).size().unstack()
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
churn_percentages.plot(
    kind="bar", stacked=True, color=["#EEBA30", "#222F5B"], figsize=(8, 5)
)
plt.title("Churn Percentage by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Grouped bar chart for Internet Service and Dependents
churn_counts = (
    telco_data.groupby(["InternetService", "Dependents", "Churn"])
    .size()
    .unstack(fill_value=0)
)
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
churn_percentages = churn_percentages["Yes"].unstack()

churn_percentages = churn_percentages.reset_index()

plot_data = churn_percentages.melt(
    id_vars=["InternetService"], var_name="Dependents", value_name="Churn Percentage"
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=plot_data,
    x="InternetService",
    y="Churn Percentage",
    hue="Dependents",
    palette={"Yes": "#cff65e", "No": "#ffa550"},
)

plt.title("Churn Percentage by Internet Service and Dependents")
plt.xlabel("Internet Service")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Dependents")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Count plot for tenure and Churn
plt.figure(figsize=(16, 8))
sns.countplot(x="tenure", hue="Churn", data=telco_data)
plt.title("Tenure Distribution by Churn Status")
plt.tight_layout()
plt.show()

# Tenure Distribution by Internet Service Type
plt.figure(figsize=(10, 6))

sns.histplot(
    data=telco_data[telco_data.InternetService == "No"],
    x="tenure",
    kde=True,
    label="No",
    color="orange",
    stat="density",
    alpha=0.3,
)

sns.histplot(
    data=telco_data[telco_data.InternetService == "DSL"],
    x="tenure",
    kde=True,
    label="DSL",
    color="blue",
    stat="density",
    alpha=0.3,
)

sns.histplot(
    data=telco_data[telco_data.InternetService == "Fiber optic"],
    x="tenure",
    kde=True,
    label="Fiber optic",
    color="green",
    stat="density",
    alpha=0.3,
)

plt.title("Tenure Distribution by Internet Service Type")
plt.xlabel("Tenure")
plt.ylabel("Density")
plt.legend(title="Internet Service Type", fontsize=10)
plt.tight_layout()
plt.show()

# KDE plot for tenure distribution by Streaming TV Subscription
sns.kdeplot(
    telco_data.tenure[telco_data.StreamingTV == "No"],
    label="No",
    color="orange",
    linewidth=2,
)
sns.kdeplot(
    telco_data.tenure[telco_data.StreamingTV == "Yes"],
    label="Yes",
    color="blue",
    linewidth=2,
)
sns.kdeplot(
    telco_data.tenure[telco_data.StreamingTV == "No internet service"],
    label="No Internet Service",
    color="green",
    linewidth=2,
)
plt.title("Tenure Distribution by Streaming TV Service Subscription")
plt.legend()
plt.tight_layout()
plt.show()

# KDE plot for tenure distribution by Streaming Movies Subscription
sns.kdeplot(
    telco_data.tenure[telco_data.StreamingMovies == "No"],
    label="No",
    color="orange",
    linewidth=2,
)
sns.kdeplot(
    telco_data.tenure[telco_data.StreamingMovies == "Yes"],
    label="Yes",
    color="blue",
    linewidth=2,
)
sns.kdeplot(
    telco_data.tenure[telco_data.StreamingMovies == "No internet service"],
    label="No Internet Service",
    color="green",
    linewidth=2,
)
plt.title("Tenure Distribution by Streaming Movies Service Subscription")
plt.legend()
plt.tight_layout()
plt.show()

# Stacked bar chart for Payment Method and Churn
payment_method_var = "PaymentMethod"
churn_counts = (
    telco_data.groupby([payment_method_var, "Churn"]).size().unstack(fill_value=0)
)
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
churn_percentages.plot(
    kind="bar", stacked=True, color=["#EEBA30", "#222F5B"], figsize=(8, 5)
)
plt.title("Churn Percentage by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Grouped bar chart: Payment Method, Senior Citizen, and Churn
churn_counts = (
    telco_data.groupby(["PaymentMethod", "SeniorCitizen", "Churn"])
    .size()
    .unstack(fill_value=0)
)
churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
churn_percentages = churn_percentages["Yes"].unstack().reset_index()
plot_data = churn_percentages.melt(
    id_vars=["PaymentMethod"], var_name="SeniorCitizen", value_name="Churn Percentage"
)
plot_data["SeniorCitizen"] = plot_data["SeniorCitizen"].map({0: "No", 1: "Yes"})
plt.figure(figsize=(10, 6))
sns.barplot(
    data=plot_data,
    x="PaymentMethod",
    y="Churn Percentage",
    hue="SeniorCitizen",
    palette={"No": "#66cccc", "Yes": "#d43d1a"},
)
plt.title("Churn Percentage by Payment Method and Senior Citizen Status")
plt.xlabel("Payment Method")
plt.ylabel("Churn Percentage (%)")
plt.legend(title="Senior Citizen")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# KDE plot for Monthly Charges by Churn
plt.figure(figsize=(10, 6))
sns.kdeplot(
    telco_data.MonthlyCharges[telco_data["Churn"] == "No"],
    color="#e95c29",
    fill=True,
    linewidth=2,
    label="No Churn",
)
sns.kdeplot(
    telco_data.MonthlyCharges[telco_data["Churn"] == "Yes"],
    color="#14293a",
    fill=True,
    linewidth=2,
    label="Churn",
)
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Monthly Charges")
plt.ylabel("Density")
plt.legend(title="Churn Status", title_fontsize=14, fontsize=12, loc="upper right")
plt.tight_layout()
plt.show()
