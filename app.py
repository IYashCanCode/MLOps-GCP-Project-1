import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load("bank_churn_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON payload
        data = request.get_json()

        # Expected fields
        expected_fields = [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ]

        # Convert input to DataFrame
        input_data = pd.DataFrame([{
            field: data[field] for field in expected_fields
        }])

        # Make prediction
        pred_proba = model.predict(input_data)[0]
        prediction = int(pred_proba > 0.5)

        return jsonify({
            "churn_probability": float(pred_proba),
            "churn_prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return "✅ Bank Churn Prediction API is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
