from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load ML model and label encoder
model = None
le = None
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("⚠️ Could not load model:", e)

# Store the latest sensor data in memory
latest_sensor_data = None

@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data
    try:
        data = request.get_json()
        
        # Check if data is received
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Extract and convert values
        latest_sensor_data = {
            "N": float(data.get("N", 0)),
            "P": float(data.get("P", 0)),
            "K": float(data.get("K", 0)),
            "rainfall": float(data.get("rainfall", 0)),
            "temperature": float(data.get("temperature", 0))
        }
        
        print(f"📡 Received sensor data: {latest_sensor_data}")
        
        return jsonify({
            "status": "success",
            "message": "Data received",
            "data": latest_sensor_data
        })
    except Exception as e:
        print(f"❌ Error in sensor_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    if not latest_sensor_data:
        return jsonify({"error": "No sensor data available"}), 400

    try:
        features = [
            latest_sensor_data["N"],
            latest_sensor_data["P"],
            latest_sensor_data["K"],
            latest_sensor_data["rainfall"],
            latest_sensor_data["temperature"]
        ]

        print(f"🤖 Predicting with features: {features}")
        
        # Predict
        prediction_index = model.predict([features])[0]
        crop_name = le.inverse_transform([prediction_index])[0]

        print(f"🌱 Recommended crop: {crop_name}")
        
        return jsonify({
            "status": "success",
            "recommended_crop": crop_name,
            "features_used": latest_sensor_data
        })
    except Exception as e:
        print(f"❌ Error in recommend_crops: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/crop-soil", methods=["GET"])
def crop_soil():
    crop = request.args.get("crop", "")
    return jsonify({"crop": crop, "info": "Ideal soil info here"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "latest_data": latest_sensor_data,
        "endpoints": [
            "POST /sensor-data",
            "GET /recommend-crops",
            "GET /crop-soil?crop=<crop_name>"
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)