import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

telco_data = pd.read_csv("Dataset/telco_data_cleaned.csv")

telco_data["Churn"] = telco_data["Churn"].map({"Yes": 1, "No": 0})

timevar = telco_data["tenure"]
eventvar = telco_data["Churn"]

# Plot overall survival curve
plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()
kmf.fit(timevar, event_observed=eventvar, label="All Customers")
kmf.plot(color="#2A4D69")
plt.ylabel("Probability of Customer Survival")
plt.xlabel("Tenure")
plt.title("Kaplan-Meier Curve: Overall Survival")
plt.tight_layout()
plt.show()

categories = [
    "Partner",
    "Dependents",
    "SeniorCitizen",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "DeviceProtection",
    "OnlineBackup",
    "StreamingMovies",
    "StreamingTV",
]

# survival analysis vs each category
for category in categories:
    print(f"\n--- Survival Analysis: {category} ---")
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)

    for group in telco_data[category].unique():
        kmf = KaplanMeierFitter()
        kmf.fit(
            timevar[telco_data[category] == group],
            event_observed=eventvar[telco_data[category] == group],
            label=str(group),
        )
        kmf.plot(ax=ax, ci_show=False)

    plt.title(f"Survival of Customers by {category}")
    plt.xlabel("Tenure")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1)
    plt.legend(title=category)
    plt.tight_layout()
    plt.show()

    # Log-rank test for the category
    if len(telco_data[category].unique()) > 1:
        logrank_results = multivariate_logrank_test(
            timevar, telco_data[category], eventvar
        )
        print(f"\nLog-Rank Test Results for {category}:")
        logrank_results.print_summary()
    else:
        print(f"Category {category} has only one unique value. Skipping log-rank test.")
