from flask import Flask, request, jsonify
import joblib
import datetime

# ---------------------------
# Load ML model
# ---------------------------
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("✅ ML Model Loaded")
except Exception as e:
    print("⚠️ Could not load ML model:", e)
    use_ml = False

# ---------------------------
# Initialize app and data storage
# ---------------------------
app = Flask(__name__)
latest_sensor_data = {}  # store latest sensor reading

# ---------------------------
# Available crops and dummy data
# ---------------------------
crops_list = ["Maize", "Mango", "Groundnuts", "Cowpeas", "Beans", "Watermelon"]
ideal_soil = {
    "Maize": {"pH": "5.5-7.0", "N": "60-120", "P": "30-60", "K": "40-80"},
    "Mango": {"pH": "5.5-7.5", "N": "50-100", "P": "20-50", "K": "30-70"},
    "Groundnuts": {"pH": "5.0-6.5", "N": "40-80", "P": "20-50", "K": "30-60"},
    "Cowpeas": {"pH": "5.5-7.0", "N": "50-90", "P": "20-50", "K": "30-70"},
    "Beans": {"pH": "6.0-7.5", "N": "50-100", "P": "20-50", "K": "30-70"},
    "Watermelon": {"pH": "6.0-7.5", "N": "50-100", "P": "20-60", "K": "30-80"}
}
fertilizer_plan = {
    "Maize": "Apply 60kg N, 30kg P, 40kg K per hectare.",
    "Mango": "Apply 50kg N, 20kg P, 30kg K per hectare.",
    "Groundnuts": "Apply 40kg N, 20kg P, 30kg K per hectare.",
    "Cowpeas": "Apply 50kg N, 20kg P, 30kg K per hectare.",
    "Beans": "Apply 50kg N, 20kg P, 30kg K per hectare.",
    "Watermelon": "Apply 50kg N, 20kg P, 30kg K per hectare."
}

# ---------------------------
# Routes
# ---------------------------

@app.route("/sensor-data", methods=["POST"])
def receive_sensor_data():
    global latest_sensor_data
    try:
        data = request.json
        # Only keep features used in model
        latest_sensor_data = {
            "N": data.get("N"),
            "P": data.get("P"),
            "K": data.get("K"),
            "ph": data.get("ph"),
            "temperature": data.get("temperature"),
            "rainfall": data.get("rainfall"),
            "timestamp": datetime.datetime.now().isoformat()
        }
        return jsonify({"status": "success", "received": latest_sensor_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not use_ml or not latest_sensor_data:
        return jsonify({"recommended_crops": []})

    try:
        # Prepare features in correct order
        features = [[
            latest_sensor_data["N"],
            latest_sensor_data["P"],
            latest_sensor_data["K"],
            latest_sensor_data["ph"],
            latest_sensor_data["temperature"],
            latest_sensor_data["rainfall"]
        ]]
        pred_index = model.predict(features)[0]
        crop_name = le.inverse_transform([pred_index])[0]
        return jsonify({"recommended_crops": [crop_name]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/crop-soil", methods=["GET"])
def crop_soil():
    crop = request.args.get("crop")
    if crop in ideal_soil:
        return jsonify(ideal_soil[crop])
    else:
        return jsonify({"error": "Crop not found"}), 404

@app.route("/fertilizer", methods=["GET"])
def fertilizer():
    crop = request.args.get("crop")
    if crop in fertilizer_plan:
        return jsonify({"plan": fertilizer_plan[crop]})
    else:
        return jsonify({"plan": "No plan available"})

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
