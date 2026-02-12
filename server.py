from flask import Flask, request, jsonify
import joblib
import traceback
import os
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
# Hardcoded ideal soil dataset
# ----------------------------
IDEAL_SOIL = {
    "beans":      {"N": [0,40],    "P": [55,80],  "K": [15,25], "temperature": [15.33,24.92], "moisture":[18.09,24.97], "pH":[5.50,6.00]},
    "cowpeas":    {"N": [0,40],    "P": [35,60],  "K": [15,25], "temperature": [27.01,29.91], "moisture":[80.03,89.99], "pH":[6.22,7.20]},
    "groundnuts": {"N": [0,40],    "P": [35,60],  "K": [15,25], "temperature": [24.02,31.99], "moisture":[40.01,64.96], "pH":[3.50,9.94]},
    "maize":      {"N": [60,100],  "P": [35,60],  "K": [15,25], "temperature": [18.04,26.55], "moisture":[55.28,74.83], "pH":[5.51,7.00]},
    "mango":      {"N": [0,40],    "P": [15,40],  "K": [25,35], "temperature": [27.00,35.99], "moisture":[45.02,54.96], "pH":[4.51,6.97]},
    "watermelon": {"N": [80,120],  "P": [5,30],   "K": [45,55], "temperature": [24.04,26.99], "moisture":[80.03,89.98], "pH":[6.00,6.96]}
}

FEATURE_NAMES = ["N","P","K","temperature","moisture","pH"]
Z_THRESHOLD = 3.0

latest_sensor_data = None
latest_recommendation = None

# ----------------------------
# Helper: Z-score validation
# ----------------------------
feature_means = {f: np.mean([IDEAL_SOIL[crop][f][0] for crop in IDEAL_SOIL]) for f in FEATURE_NAMES}
feature_stds  = {f: np.std([IDEAL_SOIL[crop][f][0] for crop in IDEAL_SOIL]) for f in FEATURE_NAMES}

def is_within_zscore(sensor_values):
    z_scores = [abs((sensor_values[f]-feature_means[f])/feature_stds[f]) if feature_stds[f]>0 else 0 for f in FEATURE_NAMES]
    return all(z <= Z_THRESHOLD for z in z_scores)

# ----------------------------
# Endpoint: Receive sensor data
# ----------------------------
@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error":"No JSON payload"}),400

        # Store sensor data
        latest_sensor_data = {f: float(data.get(f,0)) for f in FEATURE_NAMES}
        print(f"📡 Received sensor data: {latest_sensor_data}")

        if not is_within_zscore(latest_sensor_data):
            latest_recommendation = "No crop recommended (out-of-scope)"
        else:
            # ML model prediction
            if model_loaded:
                try:
                    features = [latest_sensor_data[f] for f in FEATURE_NAMES]
                    prediction_index = model.predict([features])[0]
                    latest_recommendation = le.inverse_transform([prediction_index])[0]
                except:
                    latest_recommendation = "Prediction failed"
            else:
                latest_recommendation = "maize" if latest_sensor_data["temperature"]>25 else "wheat"

        return jsonify({
            "status":"success",
            "sensor_data":latest_sensor_data,
            "recommended_crop":latest_recommendation,
            "model_used":"ML model" if model_loaded else "fallback"
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500

# ----------------------------
# Get recommended crops
# ----------------------------
@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if latest_sensor_data is None:
        return jsonify({"error":"No sensor data yet"}),404
    return jsonify({
        "recommended_crops":[latest_recommendation],
        "sensor_data":latest_sensor_data,
        "status":"success"
    })

# ----------------------------
# Ideal soil for a crop
# ----------------------------
@app.route("/ideal-soil", methods=["GET"])
def ideal_soil():
    crop = request.args.get("crop","").lower()
    if crop not in IDEAL_SOIL:
        return jsonify({"error":"Crop not found"}),404
    return jsonify({
        "crop":crop,
        "ideal_soil":{f: {"min": float(IDEAL_SOIL[crop][f][0]), "max": float(IDEAL_SOIL[crop][f][1])} for f in FEATURE_NAMES},
        "status":"success"
    })

# ----------------------------
# Fertilizer suggestions
# ----------------------------
@app.route("/fertilizer", methods=["GET"])
def fertilizer():
    crop = request.args.get("crop","").lower()
    if crop not in IDEAL_SOIL:
        return jsonify({"error":"Crop not found"}),404
    if latest_sensor_data is None:
        return jsonify({"error":"No sensor data yet"}),404

    plan = []
    for f in FEATURE_NAMES:
        val = latest_sensor_data[f]
        min_val,max_val = IDEAL_SOIL[crop][f]
        if val < min_val:
            plan.append(f"{f} low: consider increasing")
        elif val > max_val:
            plan.append(f"{f} high: consider reducing")
    if not plan:
        plan.append("Soil conditions are optimal for this crop.")

    return jsonify({
        "crop":crop,
        "sensor_data":latest_sensor_data,
        "fertilizer_suggestions":plan,
        "status":"success"
    })

# ----------------------------
# Home / status
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":"online",
        "has_sensor_data": latest_sensor_data is not None,
        "latest_recommendation": latest_recommendation,
        "endpoints":[
            "POST /sensor-data",
            "GET /recommend-crops",
            "GET /ideal-soil?crop=<name>",
            "GET /fertilizer?crop=<name>"
        ]
    })

if __name__=="__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
