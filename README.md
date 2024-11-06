## Understanding Early-Stage Customer Churn: Insights and Strategies for Retention

### Table of Contents: 
- Data Overview
- Exploratory Data Analysis (EDA)
- EDA Summary
- Modeling
- Tableau Dashboard
- Conclusion

### Data Overview: 
Dataset Attributes:

- customerID : Customer ID
- gender : Whether the customer is a male or a female
- SeniorCitizen : Whether the customer is a senior citizen or not (1, 0)
- Partner : Whether the customer has a partner or not (Yes, No)
- Dependents : Whether the customer has dependents or not (Yes, No)
- tenure : Number of months the customer has stayed with the company
- PhoneService : Whether the customer has a phone service or not (Yes, No)
- MultipleLines : Whether the customer has multiple lines or not (Yes, No, No phone service)
- InternetService : Customer’s internet service provider (DSL, Fiber optic, No)
- OnlineSecurity : Whether the customer has online security or not (Yes, No, No internet service)
- OnlineBackup : Whether the customer has online backup or not (Yes, No, No internet service)
- DeviceProtection : Whether the customer has device protection or not (Yes, No, No internet service)
- TechSupport : Whether the customer has tech support or not (Yes, No, No internet service)
- StreamingTV : Whether the customer has streaming TV or not (Yes, No, No internet service)
- StreamingMovies : Whether the customer has streaming movies or not (Yes, No, No internet service)
- Contract : The contract term of the customer (Month-to-month, One year, Two year)
- PaperlessBilling : Whether the customer has paperless billing or not (Yes, No)
- PaymentMethod : The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- MonthlyCharges : The amount charged to the customer monthly
- TotalCharges : The total amount charged to the customer
- Churn : Whether the customer churned or not (Yes or No)

*Missing value*
![alt text](graph/missing_value.png)

- There is no missing value in this dataset 

*Churn status*
![alt text](graph1/PieChart.png)
- (write)

![alt text](graph1/heatmap_mean.png)
-(write)

![alt text](<graph1/churn vs contract 1-72 months.png>)
-(write)

![alt text](<graph1/Churn Rate vs Tenure 1.png>)
-(write)

![alt text](<graph1/Churn Rate vs Tenure.png>)
-(write)

![alt text](<graph1/monthly charge vs churn.png>)
-(write)

-![alt text](<graph1/total chrages vs churn.png>)
-(write)

### EDA

*Group 1 Client infomation*
![alt text](<graph1/churn vs Dependent.png>)
![alt text](<graph1/churn vs gender.png>)
![alt text](<graph1/Churn vs Senior Citizen.png>)

*Group 2 Service subscibtion*
![alt text](<graph1/Churn vs PhoneService .png>)
![alt text](<graph1/Churn vs MultipleLine .png>)
![alt text](<graph1/Churn vs Internet Service .png>)
![alt text](<graph1/churn vs StreamingTV.png>)
![alt text](<graph1/churn vs StreamingMovies.png>)

![alt text](<graph1/Churn vs Online Security.png>)
![alt text](<graph1/Churn vs OnlineBackup.png>)
![alt text](<graph1/Churn vs DeviceProtection.png>)
![alt text](<graph1/Churn vs TechSupport.png>)

![alt text](<graph1/churn vs contract.png>)
![alt text](<graph1/churn vs paperlessBilling.png>)
![alt text](<graph1/Churn vs payment method.png>)

*Summary*
2. Churn by Internet Service
Insight: Customers with fiber optic internet experience the highest churn rate, indicating potential dissatisfaction, while customers without internet service have a much lower churn rate.
Actionable Strategy: Investigate why fiber optic users are dissatisfied (e.g., performance or cost) and address these issues by offering improved service reliability or more competitive pricing. Retaining fiber optic customers, who are likely high-revenue, should be a priority.
3. Churn by Payment Method
Insight: Customers paying via electronic check have the highest churn rates compared to those using automatic payment methods (credit card or bank transfer).
Actionable Strategy: Encourage customers to switch to automatic payment methods by offering convenience and incentives. This could improve customer retention by reducing churn among high-risk segments.
4. Churn by Tenure
Insight: Early-stage customers (1-12 months) churn at significantly higher rates, especially in their first few months. Tenure appears to be a critical factor for churn.
Actionable Strategy: Implement proactive engagement strategies during the first 12 months (e.g., onboarding support, exclusive offers) to improve early-stage customer retention. Focus on providing high-value customers with personalized service early to increase their lifetime value.
5. Churn by Online Services (Security, Backup, Tech Support)
Insight: Customers who do not subscribe to value-added services like online security, backup, or tech support have higher churn rates.
Actionable Strategy: Promote value-added services such as online security and tech support, especially to high-value customers, to enhance their service experience and improve retention.
6. Churn by Streaming Services (TV and Movies)
Insight: Customers who do not have streaming services (TV and Movies) churn more frequently than those who have these services.
Actionable Strategy: Encourage high-value customers to subscribe to streaming bundles by offering discounts or promotional offers, as entertainment services seem to enhance customer loyalty and satisfaction.
7. Churn by Dependents
Insight: Customers without dependents churn more frequently, suggesting that they might be more price-sensitive or less tied to the service compared to customers with families.
Actionable Strategy: Consider targeting high-value customers without dependents with personalized retention offers, such as flexible pricing or feature-rich packages that align with their needs.
8. Churn by Senior Citizen Status
Insight: Senior citizens have a higher churn rate compared to non-senior citizens. This demographic may face specific challenges that lead to churn (e.g., higher costs or inadequate service offerings).
Actionable Strategy: Create tailored retention strategies for senior citizens, such as offering age-friendly services or discounts, to retain these potentially high-value customers.



# Customer_Retention_Analysis
