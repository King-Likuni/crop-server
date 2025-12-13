from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Store latest sensor data here
latest_sensor_data = {}

# Load ML model
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("ML Model Loaded")
except Exception as e:
    print("Could not load ML model:", e)
    use_ml = False

# Endpoint to receive ESP data
@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400
    
    # Only keep the features your model uses
    latest_sensor_data = {
        "N": data.get("N"),
        "P": data.get("P"),
        "K": data.get("K"),
        "temperature": data.get("temperature"),
        "rainfall": data.get("rainfall"),
        "ph": data.get("ph")
    }

    return jsonify({"status": "success", "received": latest_sensor_data}), 200

# Endpoint to recommend crops
@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not latest_sensor_data:
        return jsonify({"recommended_crops": [], "message": "No sensor data yet"}), 200

    # Use your ML model
    try:
        features = [[
            latest_sensor_data["N"],
            latest_sensor_data["P"],
            latest_sensor_data["K"],
            latest_sensor_data["temperature"],
            latest_sensor_data["rainfall"],
            latest_sensor_data["ph"]
        ]]
        preds = model.predict(features)
        crops = le.inverse_transform(preds)
        return jsonify({"recommended_crops": list(crops)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
