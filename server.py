from flask import Flask, request, jsonify
import joblib
import traceback
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

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
# Load dataset for ideal soil
# ----------------------------
dataset = None
try:
    if os.path.exists("Final_crop_data.csv"):
        dataset = pd.read_csv("Final_crop_data.csv")
        print("✅ Dataset loaded successfully")
        print(f"Dataset shape: {dataset.shape}")
    else:
        print("❌ Final_crop_data.csv not found")
except Exception as e:
    print("⚠️ Could not load dataset:", e)
    print(traceback.format_exc())

# ----------------------------
# Load feature statistics for Z-score validation
# ----------------------------
FEATURE_NAMES = ["N", "P", "K", "moisture", "temperature", "pH"]
feature_means = None
feature_stds = None
Z_THRESHOLD = 3.0  # max allowed Z-score

try:
    if os.path.exists("feature_means.pkl") and os.path.exists("feature_stds.pkl"):
        feature_means = joblib.load("feature_means.pkl")
        feature_stds = joblib.load("feature_stds.pkl")
        print("✅ Feature statistics loaded for Z-score validation")
        print("Means:", feature_means)
        print("Stds :", feature_stds)
    else:
        print("❌ Feature stats not found")
except Exception as e:
    print("⚠️ Could not load feature statistics:", e)
    print(traceback.format_exc())

# ----------------------------
# Global storage for latest data
# ----------------------------
latest_sensor_data = None
latest_recommendation = None

# ----------------------------
# Helper: Check if data is within Z-score bounds
# ----------------------------
def is_within_zscore(sensor_values):
    global feature_means, feature_stds, FEATURE_NAMES, Z_THRESHOLD
    if feature_means is None or feature_stds is None:
        return True  # fallback if stats not loaded

    sensor_array = np.array([sensor_values.get(f, 0) for f in FEATURE_NAMES])
    z_scores = np.abs((sensor_array - feature_means) / feature_stds)
    print("🔍 Z-scores:", dict(zip(FEATURE_NAMES, z_scores)))
    return np.all(z_scores <= Z_THRESHOLD)

# ----------------------------
# Endpoint: Receive sensor data
# ----------------------------
@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        # Store sensor data
        latest_sensor_data = {
            "N": float(data.get("N", 0)),
            "P": float(data.get("P", 0)),
            "K": float(data.get("K", 0)),
            "moisture": float(data.get("moisture", 0)),
            "temperature": float(data.get("temperature", 0)),
            "pH": float(data.get("pH", 0)),
        }
        print(f"📡 Received sensor data: {latest_sensor_data}")

        # Z-score validation
        if not is_within_zscore(latest_sensor_data):
            latest_recommendation = "No crop recommended (out-of-scope values)"
            print("⚠️ Sensor data out of model scope!")
        else:
            # Predict recommended crop
            if model_loaded:
                try:
                    features = [latest_sensor_data[f] for f in FEATURE_NAMES]
                    print(f"🤖 Predicting with features: {features}")
                    prediction_index = model.predict([features])[0]
                    latest_recommendation = le.inverse_transform([prediction_index])[0]
                    print(f"🌱 Model predicted: {latest_recommendation}")
                except Exception as e:
                    print(f"❌ Prediction error: {e}")
                    print(traceback.format_exc())
                    latest_recommendation = "Prediction failed"
            else:
                # Fallback rule-based
                latest_recommendation = "maize" if latest_sensor_data["temperature"] > 25 else "wheat"

        return jsonify({
            "status": "success",
            "message": "Sensor data received",
            "sensor_data": latest_sensor_data,
            "recommended_crop": latest_recommendation,
            "model_used": "ML model" if model_loaded else "fallback"
        })

    except Exception as e:
        print(f"❌ Error in sensor_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Optional endpoint: Just get latest recommendation
# ----------------------------
@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if latest_sensor_data is None:
        return jsonify({"error": "No sensor data available. ESP32 needs to send data first."}), 404

    return jsonify({
        "recommended_crops": [latest_recommendation],
        "sensor_data": latest_sensor_data,
        "model_used": "ML model" if model_loaded else "fallback",
        "status": "success"
    })

# ----------------------------
# Crop soil info
# ----------------------------
@app.route("/crop-soil", methods=["GET"])
def crop_soil():
    crop = request.args.get("crop", "").lower()
    soil_info = {
        "maize": "Loamy soil, pH 5.8-7.0, well-drained",
        "mango": "Deep sandy loam, pH 5.5-7.5",
        "groundnuts": "Sandy loam, pH 5.8-6.2",
        "cowpeas": "Well-drained sandy soil, pH 5.5-6.5",
        "beans": "Loamy soil, pH 6.0-7.0",
        "watermelon": "Sandy loam, pH 6.0-6.8",
        "rice": "Clay loam, pH 5.5-6.5",
        "wheat": "Clay loam, pH 6.0-7.5"
    }
    return jsonify({
        "crop": crop,
        "soil_info": soil_info.get(crop, "Soil info not available"),
        "ideal_ph": "Check specific crop for pH requirements"
    })

# ----------------------------
# Ideal soil values from dataset
# ----------------------------
@app.route("/ideal-soil", methods=["GET"])
def ideal_soil():
    global dataset
    crop = request.args.get("crop", "").lower()
    if dataset is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    if not crop:
        return jsonify({"error": "Crop name required"}), 400

    crop_data = dataset[dataset["label"].str.lower() == crop]
    if crop_data.empty:
        return jsonify({"error": "Crop not found"}), 404

    ideal = {
        "N": {"min": float(crop_data["N"].min()), "max": float(crop_data["N"].max())},
        "P": {"min": float(crop_data["P"].min()), "max": float(crop_data["P"].max())},
        "K": {"min": float(crop_data["K"].min()), "max": float(crop_data["K"].max())},
        "temperature": {"min": float(crop_data["temperature"].min()), "max": float(crop_data["temperature"].max())},
        "moisture": {"min": float(crop_data["moisture"].min()), "max": float(crop_data["moisture"].max())},
        "pH": {"min": float(crop_data["pH"].min()), "max": float(crop_data["pH"].max())},
    }

    return jsonify({"crop": crop, "ideal_soil": ideal, "status": "success"})

# ----------------------------
# Fertilizer plans
# ----------------------------
@app.route("/fertilizer", methods=["GET"])
def fertilizer():
    global latest_sensor_data, dataset
    crop = request.args.get("crop", "").lower()
    if not crop:
        return jsonify({"error": "Crop name required"}), 400

    # Get ideal soil values
    ideal_data = dataset[dataset["label"].str.lower() == crop]
    if ideal_data.empty:
        return jsonify({"error": "Crop not found in dataset"}), 404

    if latest_sensor_data is None:
        return jsonify({"error": "No sensor data available"}), 404

    plan = []
    # Compare sensor values to ideal ranges
    for nutrient in ["N", "P", "K", "pH", "temperature", "moisture"]:
        sensor_val = latest_sensor_data.get(nutrient, None)
        ideal_min = float(ideal_data[nutrient].min())
        ideal_max = float(ideal_data[nutrient].max())
        if sensor_val is not None:
            if sensor_val < ideal_min:
                plan.append(f"{nutrient} low: consider increasing")
            elif sensor_val > ideal_max:
                plan.append(f"{nutrient} high: consider reducing")

    if not plan:
        plan.append("Soil conditions are optimal for this crop.")

    return jsonify({
        "crop": crop,
        "sensor_data": latest_sensor_data,
        "fertilizer_suggestions": plan,
        "status": "success"
    })

# ----------------------------
# Home / Status
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "has_sensor_data": latest_sensor_data is not None,
        "latest_recommendation": latest_recommendation,
        "endpoints": [
            "POST /sensor-data",
            "GET /recommend-crops",
            "GET /crop-soil?crop=<name>",
            "GET /ideal-soil?crop=<name>",
            "GET /fertilizer?crop=<name>"
        ]
    })

# ----------------------------
# Run Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)


