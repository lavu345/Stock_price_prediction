import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---------------- Load Model & Data ----------------
model = joblib.load('best_stock_model_v3.pkl')

# Feature defaults (for missing inputs)
with open('feature_defaults.json', 'r') as f:
    feature_defaults = json.load(f)

# Original dataset (for lag features)
data = pd.read_csv("C:/Users/Lavanya D R/Stock price prediction/tsla_2014_2023.csv")
data.dropna(inplace=True)

# ---------------- Home Route ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- Predict Route ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()
            features = feature_defaults.copy()

            # Update features with user input
            for key, value in form_data.items():
                if value:
                    features[key] = float(value)

            # Add lagged features from last 5 days
            for i in range(1, 6):
                features[f'close_lag_{i}'] = data['close'].iloc[-i]
                features[f'high_lag_{i}'] = data['high'].iloc[-i]
                features[f'low_lag_{i}'] = data['low'].iloc[-i]
                features[f'open_lag_{i}'] = data['open'].iloc[-i]

            # Feature vector in correct order
            feature_vector = [features[key] for key in model.feature_names_in_]

            # Make prediction
            prediction = np.expm1(model.predict([feature_vector]))

            # Render result page
            return render_template('result.html', prediction=prediction[0])
        except Exception as e:
            return render_template('result.html', error=str(e))

    # GET request → show predict form page
    return render_template('predict.html')

# ---------------- Analysis Route ----------------
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# ---------------- API Endpoint ----------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        json_data = request.get_json()
        features = feature_defaults.copy()

        # Update features with user input
        for key, value in json_data.items():
            if value:
                features[key] = float(value)

        # Add lagged features
        for i in range(1, 6):
            features[f'close_lag_{i}'] = data['close'].iloc[-i]
            features[f'high_lag_{i}'] = data['high'].iloc[-i]
            features[f'low_lag_{i}'] = data['low'].iloc[-i]
            features[f'open_lag_{i}'] = data['open'].iloc[-i]

        feature_vector = [features[key] for key in model.feature_names_in_]
        prediction = np.expm1(model.predict([feature_vector]))

        return jsonify({'predicted_next_day_close': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------------- Main ----------------
if __name__ == '__main__':
    app.run(debug=True)
