import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('Telco-Customer-Churn.csv')

# Drop 'customerID' if it exists
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Convert 'Churn' to a binary numerical value
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Handle TotalCharges: convert to numeric, filling NAs with median value if needed
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split features and target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Define function to get model metrics
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred)
    }

# Initialize dictionary to store results
results = []

# 1. Decision Tree without resampling
model_dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8)
model_dt.fit(X_train, y_train)
results.append({"Model": "Decision Tree (Original)", **get_metrics(model_dt, X_test, y_test)})

# 2. Decision Tree with SMOTEENN
sm = SMOTEENN(random_state=100)
X_resampled, y_resampled = sm.fit_resample(X, y)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=100)
model_dt_smote = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8)
model_dt_smote.fit(Xr_train, yr_train)
results.append({"Model": "Decision Tree (SMOTEENN)", **get_metrics(model_dt_smote, Xr_test, yr_test)})

# 3. Random Forest without resampling
model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_rf.fit(X_train, y_train)
results.append({"Model": "Random Forest (Original)", **get_metrics(model_rf, X_test, y_test)})

# 4. Random Forest with SMOTEENN
model_rf_smote = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_rf_smote.fit(Xr_train, yr_train)
results.append({"Model": "Random Forest (SMOTEENN)", **get_metrics(model_rf_smote, Xr_test, yr_test)})

# 5. Random Forest with PCA on SMOTEENN
pca = PCA(n_components=0.9)  # 90% variance explained
Xr_train_pca = pca.fit_transform(Xr_train)
Xr_test_pca = pca.transform(Xr_test)
model_rf_pca = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_rf_pca.fit(Xr_train_pca, yr_train)
results.append({"Model": "Random Forest (PCA & SMOTEENN)", **get_metrics(model_rf_pca, Xr_test_pca, yr_test)})

# Display results in a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
