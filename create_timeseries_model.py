
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv("C:/Users/Lavanya D R/Stock price prediction/tsla_2014_2023.csv")
data.dropna(inplace=True)

# Create lagged features
for i in range(1, 6):
    data[f'close_lag_{i}'] = data['close'].shift(i)
    data[f'high_lag_{i}'] = data['high'].shift(i)
    data[f'low_lag_{i}'] = data['low'].shift(i)
    data[f'open_lag_{i}'] = data['open'].shift(i)

data.dropna(inplace=True)

# Define features and target
features = [col for col in data.columns if col not in ['date', 'next_day_close']]
target = 'next_day_close'

X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'best_stock_model_v2.pkl')