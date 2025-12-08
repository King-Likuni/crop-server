import os
from flask import Flask, request, jsonify
import joblib

# Load ML model
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("ML Model Loaded")
except Exception as e:
    print("Could not load ML model:", e)
    use_ml = False

app = Flask(__name__)

# ... (rest of your code remains the same)

# Dynamic port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
