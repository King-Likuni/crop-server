import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# ---------------------------
# Load ML Model (Crop Recommendation)
# ---------------------------
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("✅ ML Model Loaded Successfully")
except Exception as e:
    print("⚠️ Could not load ML model:", e)
    use_ml = False

# ---------------------------
# Sensor data (updated by ESP)
# ---------------------------
SENSOR_DATA = {
    "N": 0,
    "P": 0,
    "K": 0,
    "ph": 0,
    "humidity": 0,
    "temperature": 0,
    "rainfall": 0
}

# ---------------------------
# Ideal soil data per crop
# ---------------------------
IDEAL_SOIL = {
    "Maize": {"N": 120, "P": 60, "K": 80, "ph": 6.0, "humidity": 50},
    "Mango": {"N": 30, "P": 20, "K": 25, "ph": 6.5, "humidity": 60},
    "Groundnuts": {"N": 40, "P": 30, "K": 60, "ph": 6.0, "humidity": 55},
    # Add more crops here
}

# Fertilizer conversion multipliers
FERTILIZER_CONVERSION = {
    "N": ("Urea", 0.5),
    "P": ("Superphosphate", 2),
    "K": ("Muriate of Potash", 1.5)
}

# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation API is running!"})


@app.route("/sensor-data", methods=["POST"])
def update_sensor_data():
    try:
        data = request.get_json()
        for key in SENSOR_DATA.keys():
            if key in data:
                SENSOR_DATA[key] = data[key]
        return jsonify({"status": "success", "sensor_data": SENSOR_DATA})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not use_ml:
        return jsonify({"error": "ML model not loaded"}), 500

    try:
        features = [[
            SENSOR_DATA.get("N", 0),
            SENSOR_DATA.get("P", 0),
            SENSOR_DATA.get("K", 0),
            SENSOR_DATA.get("temperature", 0),
            SENSOR_DATA.get("humidity", 0),
            SENSOR_DATA.get("ph", 0),
            SENSOR_DATA.get("rainfall", 0)
        ]]
        prediction = model.predict(features)[0]
        crop_name = le.inverse_transform([prediction])[0]
        return jsonify({"recommended_crops": [crop_name]})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/crop-soil", methods=["GET"])
def crop_soil():
    crop = request.args.get("crop")
    if not crop:
        return jsonify({"error": "Crop not specified"}), 400

    soil_info = IDEAL_SOIL.get(crop)
    if not soil_info:
        return jsonify({"error": "Crop not found"}), 404

    return jsonify(soil_info)


@app.route("/fertilizer", methods=["GET"])
def fertilizer():
    crop = request.args.get("crop")
    if not crop:
        return jsonify({"error": "Crop not specified"}), 400

    ideal = IDEAL_SOIL.get(crop)
    if not ideal:
        return jsonify({"error": "Crop not found"}), 404

    plan = {}
    for nutrient in ["N", "P", "K"]:
        deficit = max(0, ideal[nutrient] - SENSOR_DATA.get(nutrient, 0))
        fertilizer_name, multiplier = FERTILIZER_CONVERSION[nutrient]
        amount = round(deficit * multiplier, 2)
        plan[nutrient] = {
            "deficit": deficit,
            "fertilizer": fertilizer_name,
            "amount_kg_per_ha": amount
        }

    return jsonify(plan)


# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
