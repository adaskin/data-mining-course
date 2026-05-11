
# enhanced_app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Configure logging: write to file and console
logging.basicConfig(
    filename='logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
model = joblib.load('iris_model.joblib')

# Expected number of features for Iris dataset
EXPECTED_FEATURES = 4

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Parse JSON and validate existence
    data = request.get_json()
    if not data or 'features' not in data:
        logging.warning("Request missing 'features' field")
        return jsonify({'error': 'Missing features field'}), 400

    features = data['features']

    # 2. Input validation: list of 4 numeric values
    if not isinstance(features, list):
        logging.warning("Features is not a list")
        return jsonify({'error': 'Features must be a list'}), 400
    if len(features) != EXPECTED_FEATURES:
        logging.warning(f"Expected {EXPECTED_FEATURES} features, got {len(features)}")
        return jsonify({'error': f'Expected {EXPECTED_FEATURES} features, got {len(features)}'}), 400
    if not all(isinstance(x, (int, float)) for x in features):
        logging.warning("Features contain non-numeric values")
        return jsonify({'error': 'All features must be numeric'}), 400

    # 3. Make prediction and get probability
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array).tolist()
    # Get prediction probabilities for all classes
    probabilities = model.predict_proba(features_array).tolist()
    # Confidence for the predicted class (maximum probability)
    confidence = max(probabilities[0])

    # 4. Log the request and prediction
    logging.info(
        f"Request: {features} -> Prediction: {prediction[0]}, "
        f"Confidence: {confidence:.4f}"
    )

    # 5. Return prediction and confidence
    return jsonify({
        'prediction': prediction[0],
        'confidence': confidence,
        'all_probabilities': probabilities[0]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


"""
example test requests:
valid request

curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

> Returns {"prediction": 0, "confidence": 1.0, "all_probabilities": [1.0, 0.0, 0.0]}

invalid request
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5]}'

> Returns {"error": "Expected 4 features, got 2"} with status 40
"""
