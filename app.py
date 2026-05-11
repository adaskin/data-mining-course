# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model once when the app starts
model = joblib.load('iris_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()   # read JSON from the request
    # Expect a JSON like: {"features": [5.1, 3.5, 1.4, 0.2]}
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)