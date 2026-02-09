from flask import Flask, request, jsonify
import joblib
import traceback
import os

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
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
except Exception as e:
    print("⚠️ Could not load model:", e)
    print(traceback.format_exc())

# ----------------------------
# Global storage for latest data
# ----------------------------
latest_sensor_data = None
latest_recommendation = None

# ----------------------------
# Endpoint: Receive sensor data AND return recommendation
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

        # Predict recommended crop
        if model_loaded:
            try:
                features = [
                    latest_sensor_data["N"],
                    latest_sensor_data["P"],
                    latest_sensor_data["K"],
                    latest_sensor_data["moisture"],
                    latest_sensor_data["temperature"],
                    latest_sensor_data["pH"]
                ]
                print(f"🤖 Predicting with features: {features}")
                prediction_index = model.predict([features])[0]
                latest_recommendation = le.inverse_transform([prediction_index])[0]
                print(f"🌱 Model predicted: {latest_recommendation}")
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                print(traceback.format_exc())
                latest_recommendation = "Prediction failed"
        else:
            # Fallback rule-based recommendation
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
# Fertilizer plans
# ----------------------------
@app.route("/fertilizer", methods=["GET"])
def fertilizer():
    crop = request.args.get("crop", "").lower()
    fertilizer_plans = {
        "maize": "NPK 15-15-15 at planting, Urea top dressing at 6 weeks",
        "mango": "10-10-10 NPK quarterly, add compost annually",
        "groundnuts": "Phosphate at planting, no nitrogen needed",
        "cowpeas": "Low nitrogen, focus on phosphorus and potassium",
        "beans": "20-20-20 at planting, magnesium supplement",
        "watermelon": "Balanced NPK, high potassium during fruiting"
    }
    return jsonify({
        "crop": crop,
        "plan": fertilizer_plans.get(crop, "Standard NPK recommended"),
        "timing": "Apply during planting and growth stages"
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
            "GET /fertilizer?crop=<name>"
        ]
    })

# ----------------------------
# Run Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
