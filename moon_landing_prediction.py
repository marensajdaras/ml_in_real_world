import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (you can replace this with real historical moon mission data)
data = {
    'RocketType': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # 1 = Type A, 2 = Type B
    'FuelCapacity': [5000, 6000, 5500, 7000, 4500, 6500, 5000, 7500, 4000, 6000],  # in kg
    'MissionDuration': [5, 6, 7, 8, 5, 6, 7, 8, 5, 6],  # in days
    'PayloadWeight': [1000, 1200, 1100, 1300, 900, 1150, 1050, 1400, 950, 1250],  # in kg
    'Success': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]  # 1 = Success, 0 = Failure
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df.drop('Success', axis=1)  # Features: RocketType, FuelCapacity, MissionDuration, PayloadWeight
y = df['Success']  # Target: Success (1 or 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# Decision Tree Model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_decision_tree))
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_decision_tree))

# Random Forest Model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_random_forest))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_random_forest))
