from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load ML model and label encoder
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("⚠️ Could not load model:", e)
    model = None
    le = None

# Store the latest sensor data in memory
latest_sensor_data = None

@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data
    try:
        data = request.get_json()

        # Save only the features model expects
        latest_sensor_data = {
            "N": data.get("N"),
            "P": data.get("P"),
            "K": data.get("K"),
            "rainfall": data.get("rainfall"),
            "temperature": data.get("temperature")
        }

        return jsonify({"received": latest_sensor_data, "status": "success"})
    except Exception as e:
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

        # Predict
        prediction_index = model.predict([features])[0]
        crop_name = le.inverse_transform([prediction_index])[0]

        return jsonify({"recommended_crops": [crop_name]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/crop-soil", methods=["GET"])
def crop_soil():
    # You can keep your existing crop-soil logic
    crop = request.args.get("crop", "")
    return jsonify({"crop": crop, "info": "Ideal soil info here"})

if __name__ == "__main__":
    app.run(debug=True)
