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
# Dummy sensor data (replace with actual ESP readings)
# ---------------------------
SENSOR_DATA = {
    "N": 70,   # Nitrogen in kg/ha
    "P": 40,   # Phosphorus
    "K": 50,   # Potassium
    "ph": 6.5,
    "humidity": 50,
    "temperature": 25
}

# ---------------------------
# Ideal soil data per crop (for fertilizer and soil info)
# ---------------------------
IDEAL_SOIL = {
    "Maize": {"N": 120, "P": 60, "K": 80, "ph": 6.0, "humidity": 50},
    "Mango": {"N": 30, "P": 20, "K": 25, "ph": 6.5, "humidity": 60},
    "Groundnuts": {"N": 40, "P": 30, "K": 60, "ph": 6.0, "humidity": 55},
    # Add more crops here
}

# Fertilizer conversion multipliers
FERTILIZER_CONVERSION = {
    "N": ("Urea", 0.5),            # kg fertilizer per kg N deficit
    "P": ("Superphosphate", 2),    # kg fertilizer per kg P deficit
    "K": ("Muriate of Potash", 1.5) # kg fertilizer per kg K deficit
}

# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation API is running!"})


@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not use_ml:
        return jsonify({"error": "ML model not loaded"}), 500

    # Ensure sensor data is available
    if not SENSOR_DATA:
        return jsonify({"error": "No sensor data"}), 500

    # Prepare input features for ML model
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

    # Ensure sensor data exists
    if not SENSOR_DATA:
        return jsonify({"error": "No sensor data"}), 500

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
