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
Z_THRESHOLD = 3.0
CONFIDENCE_THRESHOLD = 0.60

# Physical safety limits (absolute impossible values)
PHYSICAL_LIMITS = {
    "N": (0, 200),
    "P": (0, 200),
    "K": (0, 300),
    "moisture": (0, 100),
    "temperature": (-10, 60),
    "pH": (3.0, 10.0)
}

# =========================================================
# LOAD MODEL + LABEL ENCODER
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
# LOAD FEATURE STATISTICS (for Z-score validation)
# =========================================================
feature_means = None
feature_stds = None

try:
    if os.path.exists("feature_means.pkl") and os.path.exists("feature_stds.pkl"):
        feature_means = joblib.load("feature_means.pkl")
        feature_stds = joblib.load("feature_stds.pkl")
        print("✅ Feature statistics loaded")
    else:
        print("❌ Feature stats not found")
except Exception as e:
    print("⚠️ Failed to load feature stats:", e)

# =========================================================
# TEMP MEMORY (replace with DB later if needed)
# =========================================================
latest_sensor_data = {}
latest_recommendation = None
latest_confidence = None

# =========================================================
# VALIDATION FUNCTIONS
# =========================================================

def within_physical_limits(sensor_values):
    for key, (min_val, max_val) in PHYSICAL_LIMITS.items():
        if not (min_val <= sensor_values[key] <= max_val):
            return False
    return True


def within_zscore(sensor_values):
    if feature_means is None or feature_stds is None:
        # Safer to reject if stats missing
        return False

    try:
        sensor_array = np.array([sensor_values[f] for f in FEATURE_NAMES])
        z_scores = np.abs((sensor_array - feature_means) / feature_stds)
        return np.all(z_scores <= Z_THRESHOLD)
    except:
        return False


# =========================================================
# POST: RECEIVE SENSOR DATA
# =========================================================
@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation, latest_confidence

    try:
        data = request.get_json(force=True)

        # Validate required fields
        for key in FEATURE_NAMES:
            if key not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing key: {key}"
                }), 400

        # Convert safely to float
        latest_sensor_data = {
            key: float(data[key]) for key in FEATURE_NAMES
        }

        print("📡 Sensor data:", latest_sensor_data)

        latest_confidence = None

        # -------------------------------------------------
        # 1️⃣ Physical validation
        # -------------------------------------------------
        if not within_physical_limits(latest_sensor_data):
            latest_recommendation = "No crop recommended"

        # -------------------------------------------------
        # 2️⃣ Z-score validation
        # -------------------------------------------------
        elif not within_zscore(latest_sensor_data):
            latest_recommendation = "No crop recommended"

        # -------------------------------------------------
        # 3️⃣ ML Prediction with 60% confidence threshold
        # -------------------------------------------------
        elif model_loaded:

            features = [latest_sensor_data[f] for f in FEATURE_NAMES]

            prediction_index = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]

            confidence = float(np.max(probabilities))
            latest_confidence = round(confidence, 2)

            print("🔎 Model confidence:", confidence)

            if confidence < CONFIDENCE_THRESHOLD:
                latest_recommendation = "No crop recommended"
            else:
                latest_recommendation = le.inverse_transform([prediction_index])[0]

        else:
            latest_recommendation = "No crop recommended"

        return jsonify({
            "status": "success",
            "sensor_data": latest_sensor_data,
            "recommended_crop": latest_recommendation,
            "confidence": latest_confidence
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
        "latest_recommendation": latest_recommendation
    })


# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


