import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('best_stock_model_v3.pkl')

# Create a dataframe for feature importances
feature_importance_df = pd.DataFrame({'feature': model.feature_names_in_, 'importance': model.coef_})
feature_importance_df['importance_abs'] = feature_importance_df['importance'].abs()
feature_importance_df = feature_importance_df.sort_values(by='importance_abs', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 12))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title("Feature Importance (Log-transformed Model)")
plt.tight_layout()
plt.close()