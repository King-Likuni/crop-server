from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# =========================================================
# CONFIGURATION
# =========================================================
FEATURE_NAMES = ["N", "P", "K", "moisture", "temperature", "pH"]
CONFIDENCE_THRESHOLD = 0.60  # Model confidence threshold

# Absolute physical sanity limits (extra safety)
PHYSICAL_LIMITS = {
    "N": (0, 200),
    "P": (0, 200),
    "K": (0, 300),
    "moisture": (0, 100),
    "temperature": (-10, 60),
    "pH": (3.0, 10.0)
}

# =========================================================
# LOAD MODEL + ENCODER
# =========================================================
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
    print("⚠️ Model loading failed:", e)
    print(traceback.format_exc())

# =========================================================
# LOAD FEATURE STATS (Optional)
# =========================================================
feature_means = None
feature_stds = None

try:
    if os.path.exists("feature_means.pkl") and os.path.exists("feature_stds.pkl"):
        feature_means = joblib.load("feature_means.pkl")
        feature_stds = joblib.load("feature_stds.pkl")
        print("✅ Feature statistics loaded")
except Exception as e:
    print("⚠️ Failed to load feature stats:", e)

# =========================================================
# MEMORY STORAGE (temporary)
# =========================================================
latest_sensor_data = {}
latest_recommendation = None
latest_confidence = None

# =========================================================
# VALIDATION FUNCTIONS
# =========================================================
def within_physical_limits(sensor_values):
    """Check absolute physical plausibility."""
    for key, (min_val, max_val) in PHYSICAL_LIMITS.items():
        if not (min_val <= sensor_values[key] <= max_val):
            return False
    return True

def within_zscore(sensor_values, threshold=3.0):
    """Check whether values are within z-score limits (optional)."""
    if feature_means is None or feature_stds is None:
        return True  # If stats missing, assume valid
    try:
        sensor_array = np.array([sensor_values[f] for f in FEATURE_NAMES])
        z_scores = np.abs((sensor_array - feature_means) / feature_stds)
        return np.all(z_scores <= threshold)
    except:
        return False

# =========================================================
# POST: SENSOR DATA
# =========================================================
@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation, latest_confidence

    try:
        data = request.get_json(force=True)

        # Validate required keys
        for key in FEATURE_NAMES:
            if key not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing key: {key}"
                }), 400

        # Convert to float
        latest_sensor_data = {key: float(data[key]) for key in FEATURE_NAMES}
        print("📡 Sensor data:", latest_sensor_data)

        # 1️⃣ Physical limits check
        if not within_physical_limits(latest_sensor_data):
            latest_recommendation = "No crop recommended (physically impossible values)"
            latest_confidence = 0.0

        # 2️⃣ Optional Z-score check
        elif not within_zscore(latest_sensor_data):
            latest_recommendation = "No crop recommended (out-of-training distribution)"
            latest_confidence = 0.0

        # 3️⃣ ML Prediction
        elif model_loaded:
            features = [latest_sensor_data[f] for f in FEATURE_NAMES]
            prediction_index = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            confidence = float(np.max(probabilities))
            latest_confidence = confidence

            if confidence < CONFIDENCE_THRESHOLD:
                latest_recommendation = "No crop recommended (low model confidence)"
            else:
                latest_recommendation = le.inverse_transform([prediction_index])[0]

        else:
            latest_recommendation = "Model unavailable"
            latest_confidence = 0.0

        return jsonify({
            "status": "success",
            "sensor_data": latest_sensor_data,
            "recommended_crop": latest_recommendation,
            "confidence": latest_confidence,
            "model_loaded": model_loaded
        })

    except Exception as e:
        print("❌ Error:", e)
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =========================================================
# GET: LATEST RECOMMENDATION
# =========================================================
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
        "confidence": latest_confidence,
        "sensor_data": latest_sensor_data
    })

# =========================================================
# HEALTH CHECK
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "feature_stats_loaded": feature_means is not None,
        "latest_recommendation": latest_recommendation,
        "latest_confidence": latest_confidence
    })

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
