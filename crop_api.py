from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# ---------------------------
# Load ML model + label encoder (if available)
# ---------------------------
use_ml = False
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("✅ ML model loaded")
except Exception as e:
    print("⚠️ ML model not found, using rule-based fallback", e)

# ---------------------------
# Safe limits
# ---------------------------
safe_limits = {
    'N': (0, 130),
    'P': (0, 90),
    'K': (10, 60),
    'temperature': (10, 40),
    'rainfall': (0, 500)
}

def validate_crop_inputs(values: dict):
    for feature, (min_val, max_val) in safe_limits.items():
        if feature not in values:
            return False, f"Missing feature: {feature}"
        value = values[feature]
        if not (min_val <= value <= max_val):
            return False, f"{feature} = {value} is out of safe range ({min_val}-{max_val})"
    return True, "Inputs OK"

# ---------------------------
# Rule-based crop recommendation
# ---------------------------
def rule_based_recommendation(N, P, K, temperature, rainfall):
    if N > 60 and P > 40 and K > 30 and temperature > 25 and rainfall > 120:
        return "Maize"
    elif N < 40 and P > 50 and K > 20 and rainfall > 150:
        return "Beans"
    elif temperature > 30 and rainfall < 80:
        return "Mango"
    elif N < 30 and rainfall < 100:
        return "Cowpeas"
    elif N > 50 and P > 50 and K > 30 and rainfall > 200:
        return "Watermelon"
    else:
        return "Groundnuts"

# ---------------------------
# Ideal soil (ranges)
# ---------------------------
ideal_soil = {
    "Maize": {"N": (70,90), "P": (35,45), "K": (50,65), "temperature": (24,28), "rainfall": (100,140), "pH": (6.3,6.8)},
    "Groundnuts": {"N": (20,30), "P": (25,35), "K": (25,35), "temperature": (26,30), "rainfall": (90,120), "pH": (5.8,6.3)},
    "Cowpeas": {"N": (15,25), "P": (20,30), "K": (15,25), "temperature": (28,32), "rainfall": (80,100), "pH": (5.8,6.2)},
    "Beans": {"N": (35,45), "P": (45,55), "K": (25,35), "temperature": (20,25), "rainfall": (140,160), "pH": (5.8,6.2)},
    "Watermelon": {"N": (50,70), "P": (50,60), "K": (35,45), "temperature": (30,34), "rainfall": (180,220), "pH": (6.2,6.8)},
    "Mango": {"N": (45,55), "P": (35,45), "K": (30,40), "temperature": (32,35), "rainfall": (60,80), "pH": (5.8,6.2)}
}

# ---------------------------
# Fertilizer info
# ---------------------------
FERTILIZER_INFO = {"Urea":{"nutrient":"N","pct_nutrient":0.46},
                   "DAP":{"nutrient":"P","pct_nutrient":0.18},
                   "MOP":{"nutrient":"K","pct_nutrient":0.50}}

def fertilizer_amounts_from_deficit(deficit_dict):
    rec = {}
    n_def = deficit_dict.get("N_needed",0)
    p_def = deficit_dict.get("P_needed",0)
    k_def = deficit_dict.get("K_needed",0)
    rec["Urea_kg_per_ha"] = round(n_def/FERTILIZER_INFO["Urea"]["pct_nutrient"],1) if n_def>0 else 0.0
    rec["DAP_kg_per_ha"] = round(p_def/FERTILIZER_INFO["DAP"]["pct_nutrient"],1) if p_def>0 else 0.0
    rec["MOP_kg_per_ha"] = round(k_def/FERTILIZER_INFO["MOP"]["pct_nutrient"],1) if k_def>0 else 0.0
    return rec

def fertilizer_recommendation(crop, readings):
    crop = crop.capitalize()
    if crop not in ideal_soil:
        return None
    ideal = ideal_soil[crop]
    deficits = {}
    for nutrient in ["N","P","K","temperature","rainfall","pH"]:
        if nutrient in ideal:
            min_val,max_val = ideal[nutrient]
            value = readings.get(nutrient,0)
            deficits[f"{nutrient}_needed"] = round(max(0,min_val - value),2)
    ferts = fertilizer_amounts_from_deficit(deficits)
    return {"deficits": deficits, "fertilizers": ferts}

# ---------------------------
# ML prediction
# ---------------------------
def ml_predict_crop(readings):
    if not use_ml:
        return None
    try:
        features = [[readings['N'],readings['P'],readings['K'],readings['temperature'],readings['rainfall']]]
        pred = model.predict(features)
        crop = le.inverse_transform(pred)[0] if le else str(pred[0])
        return crop
    except:
        return None

# ---------------------------
# Simulated sensor readings
# ---------------------------
def simulated_sensor_readings():
    return {"N":45,"P":35,"K":28,"temperature":27,"rainfall":90,"pH":6.0}

# ---------------------------
# API endpoints
# ---------------------------
@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    readings = data.get("readings", simulated_sensor_readings())
    valid, msg = validate_crop_inputs(readings)
    if not valid:
        return jsonify({"error": msg}),400
    crop = ml_predict_crop(readings)
    if not crop:
        crop = rule_based_recommendation(readings['N'],readings['P'],readings['K'],readings['temperature'],readings['rainfall'])
    return jsonify({"recommended_crop": crop})

@app.route('/ideal_soil', methods=['POST'])
def ideal_soil_endpoint():
    data = request.get_json()
    crop = data.get("crop","").capitalize()
    if crop not in ideal_soil:
        return jsonify({"error":f"No data for crop '{crop}'"}),400
    return jsonify({"ideal_soil": ideal_soil[crop]})

@app.route('/fertilizer_plan', methods=['POST'])
def fertilizer_plan_endpoint():
    data = request.get_json()
    crop = data.get("crop","").capitalize()
    readings = data.get("readings", simulated_sensor_readings())
    rec = fertilizer_recommendation(crop, readings)
    if rec is None:
        return jsonify({"error":f"No data for crop '{crop}'"}),400
    return jsonify(rec)

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
