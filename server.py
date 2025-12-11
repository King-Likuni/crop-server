import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Load ML Model
# ---------------------------
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    logging.info("✅ ML Model Loaded Successfully")

    # Optional: Warm up the model for faster first prediction
    model.predict([[0, 0, 0, 0, 0, 0, 0]])
except Exception as e:
    logging.error(f"⚠️ Could not load ML model: {e}")
    use_ml = False

# ---------------------------
# Flask App
# ---------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# ---------------------------
# Home Route
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation API is running!"})

# ---------------------------
# Prediction Route
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if not use_ml:
        return jsonify({"error": "ML model not loaded"}), 500

    data = request.get_json()
    logging.info(f"Received data: {data}")

    # Expected input fields
    required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # Check missing fields
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Validate and convert inputs to float
    try:
        input_features = [[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]]
    except ValueError:
        return jsonify({"error": "All input fields must be numeric"}), 400

    # Make prediction
    try:
        prediction = model.predict(input_features)[0]
        crop_name = le.inverse_transform([prediction])[0]
        logging.info(f"Prediction: {crop_name}")
        return jsonify({"recommended_crop": crop_name})
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
