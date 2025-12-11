from flask import Flask, request, jsonify
import random
import joblib

app = Flask(__name__)

# -------------------------
# Load ML model for crop recommendation
# -------------------------
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("ML Model Loaded")
except Exception as e:
    print("Could not load ML model:", e)
    use_ml = False

# -------------------------
# Rule-based Ideal Soil Info
# -------------------------
ideal_soil_info = {
    "Maize": {"pH": "6-7", "Moisture": "Medium", "Temperature": "20-30°C"},
    "Tomatoes": {"pH": "6-7", "Moisture": "High", "Temperature": "18-28°C"},
    "Beans": {"pH": "6-7", "Moisture": "Medium", "Temperature": "18-25°C"},
    "Mango": {"pH": "5.5-7.5", "Moisture": "Medium", "Temperature": "24-30°C"},
    "Groundnuts": {"pH": "6-6.5", "Moisture": "Low-Medium", "Temperature": "25-35°C"},
    "Cowpeas": {"pH": "5.8-7", "Moisture": "Medium", "Temperature": "20-30°C"},
    "Watermelon": {"pH": "6-7", "Moisture": "High", "Temperature": "25-32°C"},
}

# -------------------------
# Mock sensor data
# -------------------------
def generate_sensor_data():
    return {
        "N": random.randint(10, 50),
        "P": random.randint(5, 30),
        "K": random.randint(10, 50),
        "temperature": random.uniform(18, 35),
        "humidity": random.uniform(40, 80),
        "ph": random.uniform(5.5, 7.5),
        "rainfall": random.uniform(0, 100)
    }

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation Server is running!"})

@app.route("/crop-soil", methods=["GET"])
def crop_soil():
    crop = request.args.get("crop")
    info = ideal_soil_info.get(crop, {"message": "No data available"})
    return jsonify(info)

@app.route("/sensor-data", methods=["GET"])
def sensor_data():
    data = generate_sensor_data()
    return jsonify(data)

@app.route("/fertilizer", methods=["GET"])
def fertilizer():
    crop = request.args.get("crop")
    # For simplicity, just use mock sensor data
    sensor = generate_sensor_data()
    # Example fertilizer plan based on crop and sensor (rule-based)
    plan = f"Fertilizer for {crop}: N={sensor['N']} P={sensor['P']} K={sensor['K']}"
    return jsonify({"plan": plan})

@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not use_ml:
        return jsonify({"error": "ML model not loaded"}), 500

    # For now, use mock sensor data as input to ML model
    data = generate_sensor_data()
    features = [[
        data["N"], data["P"], data["K"],
        data["temperature"], data["humidity"], data["ph"], data["rainfall"]
    ]]
    prediction = model.predict(features)[0]
    crop_name = le.inverse_transform([prediction])[0]
    return jsonify({"recommended_crops": [crop_name]})

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
