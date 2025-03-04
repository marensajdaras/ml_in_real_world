import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (you can replace this with a real dataset)
data = {
    'Temperature': [25, 28, 30, 22, 20, 18, 15, 10, 5, 12],
    'Humidity': [60, 65, 70, 75, 80, 85, 90, 95, 100, 55],
    'WindSpeed': [10, 12, 15, 8, 5, 7, 20, 25, 30, 9],
    'Rain': [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]  # 0 = No Rain, 1 = Rain
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df.drop('Rain', axis=1)  # Features: Temperature, Humidity, WindSpeed
y = df['Rain']  # Target: Rain (0 or 1)

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
