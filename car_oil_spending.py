import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (you can replace this with a real dataset)
data = {
    'Mileage': [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
    'CarAge': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'EngineSize': [1.6, 2.0, 1.8, 2.5, 1.4, 1.6, 2.0, 2.2, 1.8, 2.5],
    'OilSpending': [200, 250, 300, 350, 400, 450, 500, 550, 600, 650]  # Target: Oil spending in dollars
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df.drop('OilSpending', axis=1)  # Features: Mileage, CarAge, EngineSize
y = df['OilSpending']  # Target: OilSpending

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear_reg = linear_reg.predict(X_test)
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_linear_reg)))
print("Linear Regression R^2 Score:", r2_score(y_test, y_pred_linear_reg))

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_decision_tree)))
print("Decision Tree R^2 Score:", r2_score(y_test, y_pred_decision_tree))

# Random Forest Regressor
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_random_forest)))
print("Random Forest R^2 Score:", r2_score(y_test, y_pred_random_forest))
