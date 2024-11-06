# Understanding Early-Stage Customer Churn: Insights and Strategies for Retention

## Table of Contents
- [Project Overview](#project-overview)
- [Data Overview](#data-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Tableau Dashboard](#tableau-dashboard)
- [Conclusion](#conclusion)

## Project Overview
This project aims to identify and analyze early-stage customer churn patterns and develop actionable strategies to enhance customer retention. By exploring various factors influencing churn, the project provides insights to support targeted retention strategies and optimize customer engagement efforts.

## Data Overview
**Dataset Attributes:**
- **customerID**: Customer ID
- **gender**: Gender of the customer (Male, Female)
- **SeniorCitizen**: Senior citizen status (1 = Yes, 0 = No)
- **Partner**: Whether the customer has a partner (Yes, No)
- **Dependents**: Whether the customer has dependents (Yes, No)
- **tenure**: Months the customer has been with the company
- **PhoneService**: Phone service subscription (Yes, No)
- **MultipleLines**: Multiple lines subscription (Yes, No, No phone service)
- **InternetService**: Internet service provider (DSL, Fiber optic, No)
- **OnlineSecurity**: Online security subscription (Yes, No, No internet service)
- **OnlineBackup**: Online backup subscription (Yes, No, No internet service)
- **DeviceProtection**: Device protection subscription (Yes, No, No internet service)
- **TechSupport**: Tech support subscription (Yes, No, No internet service)
- **StreamingTV**: Streaming TV subscription (Yes, No, No internet service)
- **StreamingMovies**: Streaming movies subscription (Yes, No, No internet service)
- **Contract**: Contract term (Month-to-month, One year, Two year)
- **PaperlessBilling**: Paperless billing (Yes, No)
- **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)
- **MonthlyCharges**: Monthly charges
- **TotalCharges**: Total amount charged
- **Churn**: Churn status (Yes, No)

**Missing Values**: No missing values in the dataset.

## Exploratory Data Analysis (EDA)
**Key Insights and Strategies:**

1. **Internet Service and Churn**
   - **Insight**: Highest churn rate among fiber optic users; customers without internet service show lower churn.
   - **Strategy**: Investigate dissatisfaction factors for fiber optic customers and enhance service quality or pricing to retain high-value clients.

2. **Payment Method and Churn**
   - **Insight**: Customers using electronic checks have higher churn compared to automatic payment users.
   - **Strategy**: Encourage a switch to automatic payments by offering incentives to reduce churn risk.

3. **Tenure and Churn**
   - **Insight**: Customers in their first 12 months have the highest churn, particularly in the initial months.
   - **Strategy**: Implement strong onboarding and personalized engagement within the first year to increase retention.

4. **Online Services (Security, Backup, Tech Support)**
   - **Insight**: Lack of value-added services correlates with higher churn.
   - **Strategy**: Promote value-added services like online security to increase customer retention.

5. **Streaming Services (TV and Movies)**
   - **Insight**: Customers without streaming services have higher churn rates.
   - **Strategy**: Offer bundled discounts to encourage adoption of streaming services, enhancing customer loyalty.

6. **Dependents and Churn**
   - **Insight**: Customers without dependents tend to churn more, possibly due to price sensitivity.
   - **Strategy**: Target offers to price-sensitive customers without dependents, focusing on flexible and feature-rich options.

7. **Senior Citizen Status and Churn**
   - **Insight**: Senior citizens churn more frequently, possibly due to cost or service fit.
   - **Strategy**: Implement age-friendly offers or discounts to meet the specific needs of senior customers.

## Modeling
- Built and evaluated churn prediction models using Decision Tree and Random Forest classifiers.
- Applied **SMOTEENN** for data balancing and **PCA** for dimensionality reduction to enhance model accuracy.

## Tableau Dashboard
- Developed a Tableau dashboard to visualize churn patterns and key factors affecting churn, enabling stakeholders to easily interpret findings and apply strategies.

## Conclusion
This analysis reveals critical factors impacting early-stage customer churn and provides actionable strategies for improving customer retention. By understanding and addressing the factors associated with churn, organizations can create targeted initiatives to enhance customer satisfaction and reduce churn rates.

---


