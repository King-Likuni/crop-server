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

# Store latest data
latest_sensor_data = None
latest_recommendation = None

@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation
    try:
        data = request.get_json()
        
        # Extract only sensor data
        latest_sensor_data = {
            "N": float(data.get("N", 0)),
            "P": float(data.get("P", 0)),
            "K": float(data.get("K", 0)),
            "rainfall": float(data.get("rainfall", 0)),
            "temperature": float(data.get("temperature", 0))
        }
        
        print(f"📡 Received sensor data: {latest_sensor_data}")
        
        # Use model to predict crop (if model is loaded)
        if model and le:
            features = [
                latest_sensor_data["N"],
                latest_sensor_data["P"],
                latest_sensor_data["K"],
                latest_sensor_data["rainfall"],
                latest_sensor_data["temperature"]
            ]
            
            print(f"🤖 Predicting with features: {features}")
            
            try:
                prediction_index = model.predict([features])[0]
                crop_name = le.inverse_transform([prediction_index])[0]
                latest_recommendation = crop_name
                print(f"🌱 Model predicted: {crop_name}")
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                latest_recommendation = "Prediction failed"
        else:
            latest_recommendation = "Model not loaded"
        
        # Return only confirmation, NOT the recommendation
        return jsonify({
            "status": "success",
            "message": "Sensor data received",
            "data_received": {
                "N": latest_sensor_data["N"],
                "P": latest_sensor_data["P"],
                "K": latest_sensor_data["K"],
                "rainfall": latest_sensor_data["rainfall"],
                "temperature": latest_sensor_data["temperature"]
            }
            # Note: We don't return recommendation here
        })
    except Exception as e:
        print(f"❌ Error in sensor_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    """Get the crop recommendation based on latest sensor data"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    if not latest_sensor_data:
        return jsonify({"error": "No sensor data available"}), 404
    
    if not latest_recommendation:
        # Make prediction now
        try:
            features = [
                latest_sensor_data["N"],
                latest_sensor_data["P"],
                latest_sensor_data["K"],
                latest_sensor_data["rainfall"],
                latest_sensor_data["temperature"]
            ]
            prediction_index = model.predict([features])[0]
            latest_recommendation = le.inverse_transform([prediction_index])[0]
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    return jsonify({
        "recommended_crops": [latest_recommendation],
        "sensor_data": latest_sensor_data,
        "prediction_time": "latest"  # or add timestamp
    })

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

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "has_sensor_data": latest_sensor_data is not None,
        "latest_recommendation": latest_recommendation,
        "endpoints": [
            "POST /sensor-data - Send sensor data (N, P, K, rainfall, temperature)",
            "GET /recommend-crops - Get crop recommendation",
            "GET /crop-soil?crop=<name> - Get soil info",
            "GET /fertilizer?crop=<name> - Get fertilizer plan"
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)