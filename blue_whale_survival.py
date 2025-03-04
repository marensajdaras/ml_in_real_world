import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (you can replace this with real ecological data)
data = {
    'WaterTemperature': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # in Celsius
    'PollutionLevel': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # on a scale of 1-20
    'FoodAvailability': [100, 120, 110, 130, 90, 115, 105, 140, 95, 125],  # in tons of krill
    'PopulationDensity': [10, 12, 15, 8, 5, 7, 20, 25, 30, 9],  # whales per 1000 sq km
    'SurvivalChance': [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]  # survival probability (0 to 1)
}

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df.drop('SurvivalChance', axis=1)  # Features: WaterTemperature, PollutionLevel, FoodAvailability, PopulationDensity
y = df['SurvivalChance']  # Target: SurvivalChance (0 to 1)

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
