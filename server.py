import os
from flask import Flask, request, jsonify
import joblib

# Load ML model
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("ML Model Loaded Successfully")
except Exception as e:
    print("Could not load ML model:", e)
    use_ml = False

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if not use_ml:
        return jsonify({"error": "ML model not loaded"}), 500

    data = request.get_json()
    required_fields = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    input_features = [[
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"])
    ]]

    prediction = model.predict(input_features)[0]
    crop_name = le.inverse_transform([prediction])[0]

    return jsonify({"recommended_crop": crop_name})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render's port or 5000 locally
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
