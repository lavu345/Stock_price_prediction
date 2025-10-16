
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# 1. Load and Clean Data
try:
    data = pd.read_csv("C:/Users/Lavanya D R/Stock price prediction/tsla_2014_2023.csv")
except FileNotFoundError:
    print("Error: The file 'tsla_2014_2023.csv' was not found in the 'ML Projects' directory.")
    exit()

data.dropna(inplace=True)

# 2. Data Preprocessing
features = data.drop(columns=['date', 'next_day_close'])
target = data['next_day_close']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "XGBoost Regressor": XGBRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
  
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R2 Score": r2}
    print(f"{name}:")
    print(f"  RMSE: {rmse}")
    print(f"  R2 Score: {r2}")

# 4. Plotting
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs. Predicted Values ({best_model_name})")
plt.close()
print("Actual vs. Predicted plot saved.")

# Feature importance for Random Forest and XGBoost
for name in ["Random Forest Regressor", "XGBoost Regressor"]:
    model = models[name]
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = features.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title(f"Feature Importance ({name})")
        plt.tight_layout()
        plt.close()
        print(f"Feature importance plot for {name} saved.")

# Correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(data.drop(columns=['date']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.close()
print("Correlation heatmap saved.")

# 5. Print Top 5 Most Important Features
if 'Random Forest Regressor' in models:
    importances = models['Random Forest Regressor'].feature_importances_
    feature_names = features.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    print("Top 5 features from Random Forest:")
    print(feature_importance_df.head())

# 6. Save the Best-Performing Model
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = models[best_model_name]
joblib.dump(best_model, "C:/Users/Lavanya D R/ML Projects/best_stock_model.pkl")
print(f"Best model ({best_model_name}) saved as 'best_stock_model.pkl'.")

# 7. Predicting
last_row = features.iloc[-1:].values
last_row_scaled = scaler.transform(last_row)
next_day_prediction = best_model.predict(last_row_scaled)
print(f"Predicted next_day_close for the last row: {next_day_prediction[0]}")
