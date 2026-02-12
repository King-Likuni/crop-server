from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow app to access API

# ----------------------------
# Load ML model and label encoder
# ----------------------------
model = None
le = None
model_loaded = False

try:
    if os.path.exists("crop_recommendation_model.pkl") and os.path.exists("label_encoder.pkl"):
        model = joblib.load("crop_recommendation_model.pkl")
        le = joblib.load("label_encoder.pkl")
        model_loaded = True
        print("✅ Model loaded successfully")
    else:
        print("❌ Model files not found")
except Exception as e:
    print("⚠️ Could not load model:", e)
    print(traceback.format_exc())

# ----------------------------
# Load dataset
# ----------------------------
dataset = None
try:
    if os.path.exists("Final_crop_data.csv"):
        dataset = pd.read_csv("Final_crop_data.csv")
        print("✅ Dataset loaded successfully")
    else:
        print("❌ Final_crop_data.csv not found")
except Exception as e:
    print("⚠️ Could not load dataset:", e)
    print(traceback.format_exc())

# ----------------------------
# Feature configuration
# ----------------------------
FEATURE_NAMES = ["N", "P", "K", "moisture", "temperature", "pH"]
Z_THRESHOLD = 3.0

feature_means = None
feature_stds = None

try:
    if os.path.exists("feature_means.pkl") and os.path.exists("feature_stds.pkl"):
        feature_means = joblib.load("feature_means.pkl")
        feature_stds = joblib.load("feature_stds.pkl")
        print("✅ Feature statistics loaded")
except:
    print("⚠️ Feature stats not found")

# ----------------------------
# Temporary storage (replace with DB later)
# ----------------------------
latest_sensor_data = {}
latest_recommendation = None

# ----------------------------
# Z-score validation
# ----------------------------
def is_within_zscore(sensor_values):
    if feature_means is None or feature_stds is None:
        return True

    try:
        sensor_array = np.array([sensor_values[f] for f in FEATURE_NAMES])
        z_scores = np.abs((sensor_array - feature_means) / feature_stds)
        return np.all(z_scores <= Z_THRESHOLD)
    except:
        return False

# ----------------------------
# POST: Receive sensor data
# ----------------------------
@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation

    try:
        data = request.get_json(force=True)

        # Validate required keys
        for key in FEATURE_NAMES:
            if key not in data:
                return jsonify({"status": "error", "message": f"Missing key: {key}"}), 400

        # Convert to float safely
        latest_sensor_data = {
            key: float(data[key]) for key in FEATURE_NAMES
        }

        print("📡 Sensor data:", latest_sensor_data)

        # Z-score validation
        if not is_within_zscore(latest_sensor_data):
            latest_recommendation = "No crop recommended (out-of-range values)"
        else:
            if model_loaded:
                features = [latest_sensor_data[f] for f in FEATURE_NAMES]
                prediction_index = model.predict([features])[0]
                latest_recommendation = le.inverse_transform([prediction_index])[0]
            else:
                latest_recommendation = "maize" if latest_sensor_data["temperature"] > 25 else "wheat"

        return jsonify({
            "status": "success",
            "sensor_data": latest_sensor_data,
            "recommended_crop": latest_recommendation,
            "model_used": "ML" if model_loaded else "fallback"
        })

    except Exception as e:
        print("❌ Error:", e)
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------
# GET: Recommendation
# ----------------------------
@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not latest_sensor_data:
        return jsonify({
            "status": "error",
            "message": "No sensor data received yet"
        }), 404

    return jsonify({
        "status": "success",
        "recommended_crop": latest_recommendation,
        "sensor_data": latest_sensor_data
    })


# ----------------------------
# GET: Ideal soil
# ----------------------------
@app.route("/ideal-soil", methods=["GET"])
def ideal_soil():
    if dataset is None:
        return jsonify({"status": "error", "message": "Dataset not loaded"}), 500

    crop = request.args.get("crop", "").lower()
    if not crop:
        return jsonify({"status": "error", "message": "Crop required"}), 400

    crop_data = dataset[dataset["label"].str.lower() == crop]

    if crop_data.empty:
        return jsonify({"status": "error", "message": "Crop not found"}), 404

    ideal = {
        feature: {
            "min": float(crop_data[feature].min()),
            "max": float(crop_data[feature].max())
        } for feature in FEATURE_NAMES
    }

    return jsonify({
        "status": "success",
        "crop": crop,
        "ideal_soil": ideal
    })


# ----------------------------
# GET: Home
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "has_sensor_data": bool(latest_sensor_data),
        "latest_recommendation": latest_recommendation
    })


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
