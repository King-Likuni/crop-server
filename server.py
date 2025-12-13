from flask import Flask, request, jsonify
import joblib
import traceback

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
    print(traceback.format_exc())

# Store latest data
latest_sensor_data = None
latest_recommendation = None

@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation
    try:
        data = request.get_json()
        
        # Store only sensor data
        latest_sensor_data = {
            "N": float(data.get("N", 0)),
            "P": float(data.get("P", 0)),
            "K": float(data.get("K", 0)),
            "rainfall": float(data.get("rainfall", 0)),
            "temperature": float(data.get("temperature", 0))
        }
        
        print(f"📡 Received sensor data: {latest_sensor_data}")
        
        # Make prediction but don't send it back
        if model is not None and le is not None:
            try:
                features = [
                    latest_sensor_data["N"],
                    latest_sensor_data["P"],
                    latest_sensor_data["K"],
                    latest_sensor_data["rainfall"],
                    latest_sensor_data["temperature"]
                ]
                print(f"🤖 Predicting with features: {features}")
                prediction_index = model.predict([features])[0]
                crop_name = le.inverse_transform([prediction_index])[0]
                latest_recommendation = crop_name
                print(f"🌱 Model predicted: {crop_name}")
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                print(traceback.format_exc())
                latest_recommendation = None
        
        # Return ONLY confirmation, NOT recommendation
        return jsonify({
            "status": "success",
            "message": "Sensor data received",
            "data": latest_sensor_data
        })
    except Exception as e:
        print(f"❌ Error in sensor_data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    try:
        print(f"🔍 Recommend crops called. Model: {model is not None}, Data: {latest_sensor_data}")
        
        if model is None:
            return jsonify({"error": "Model not loaded on server"}), 500
        
        if latest_sensor_data is None:
            return jsonify({"error": "No sensor data available. ESP32 needs to send data first."}), 404
        
        # Make sure we have a recommendation
        if latest_recommendation is None:
            try:
                features = [
                    latest_sensor_data["N"],
                    latest_sensor_data["P"],
                    latest_sensor_data["K"],
                    latest_sensor_data["rainfall"],
                    latest_sensor_data["temperature"]
                ]
                print(f"🤖 Making new prediction with: {features}")
                prediction_index = model.predict([features])[0]
                latest_recommendation = le.inverse_transform([prediction_index])[0]
                print(f"🌱 New prediction: {latest_recommendation}")
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                print(traceback.format_exc())
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
        
        return jsonify({
            "recommended_crops": [latest_recommendation],
            "sensor_data": latest_sensor_data,
            "status": "success"
        })
    except Exception as e:
        print(f"❌ Error in recommend_crops: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

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
            "POST /sensor-data",
            "GET /recommend-crops",
            "GET /crop-soil?crop=<name>",
            "GET /fertilizer?crop=<name>"
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)