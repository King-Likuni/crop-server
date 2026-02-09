"""
crop_tool.py

- Load trained crop recommendation model (optional)
- Read sensor readings (from file or manual entry)
- Recommend crop (ML or rule-based fallback)
- Show ideal soil requirements for a crop
- Compare current soil -> compute deficits -> suggest fertilizer kg/ha
"""

import json
import joblib
import os
import math

# ---------------------------
# 0. Model Loading (optional)
# ---------------------------
use_ml = False
model = None
le = None

try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("✅ ML Model + label encoder loaded.")
except Exception as e:
    print("⚠️ Could not load ML model/label encoder (continuing without ML):", e)
    use_ml = False

# ---------------------------
# 1. Safe limits & Rule-based fallback
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
            values[feature] = 0  # auto-fill missing nutrient with 0
        value = values[feature]
        if not (min_val <= value <= max_val):
            return False, f"⚠️ {feature} = {value} is out of safe range ({min_val}-{max_val})"
    return True, "✅ Inputs OK"

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
# 2. Ideal soil database
# ---------------------------
ideal_soil = {
    "Maize":      {"N": 80, "P": 40, "K": 60, "temperature": 26, "rainfall": 120, "pH": 6.5},
    "Groundnuts": {"N": 25, "P": 30, "K": 30, "temperature": 28, "rainfall": 100, "pH": 6.0},
    "Cowpeas":    {"N": 20, "P": 25, "K": 20, "temperature": 30, "rainfall": 90,  "pH": 6.0},
    "Beans":      {"N": 40, "P": 50, "K": 30, "temperature": 22, "rainfall": 150, "pH": 6.0},
    "Watermelon": {"N": 60, "P": 55, "K": 40, "temperature": 32, "rainfall": 200, "pH": 6.5},
    "Mango":      {"N": 50, "P": 40, "K": 35, "temperature": 33, "rainfall": 70,  "pH": 6.0},
}

# ---------------------------
# 3. Fertilizer helpers
# ---------------------------
FERTILIZER_INFO = {
    "Urea":    {"nutrient": "N", "pct_nutrient": 0.46},
    "DAP":     {"nutrient": "P", "pct_nutrient": 0.18},
    "MOP":     {"nutrient": "K", "pct_nutrient": 0.50},
}

def fertilizer_amounts_from_deficit(deficit_dict):
    rec = {}
    n_def = deficit_dict.get("N_needed", 0)
    rec["Urea_kg_per_ha"] = round(n_def / FERTILIZER_INFO["Urea"]["pct_nutrient"], 1) if n_def > 0 else 0.0

    p_def = deficit_dict.get("P_needed", 0)
    rec["DAP_kg_per_ha"] = round(p_def / FERTILIZER_INFO["DAP"]["pct_nutrient"], 1) if p_def > 0 else 0.0

    k_def = deficit_dict.get("K_needed", 0)
    rec["MOP_kg_per_ha"] = round(k_def / FERTILIZER_INFO["MOP"]["pct_nutrient"], 1) if k_def > 0 else 0.0
    return rec

def fertilizer_recommendation(crop, readings):
    crop = crop.capitalize()
    if crop not in ideal_soil:
        return None
    ideal = ideal_soil[crop]

    deficits = {
        "N_needed": round(ideal.get("N",0) - readings.get("N",0),2),
        "P_needed": round(ideal.get("P",0) - readings.get("P",0),2),
        "K_needed": round(ideal.get("K",0) - readings.get("K",0),2),
        "temp_gap": round(ideal.get("temperature",0) - readings.get("temperature",0),2),
        "rain_gap": round(ideal.get("rainfall",0) - readings.get("rainfall",0),2),
        "pH_gap": round(ideal.get("pH",0) - readings.get("pH",0),2) if "pH" in readings else None
    }

    ferts = fertilizer_amounts_from_deficit(deficits)
    return {"deficits": deficits, "fertilizers": ferts}

def print_fertilizer_plan(crop, rec):
    if rec is None:
        print("❌ No data for this crop.")
        return
    deficits = rec["deficits"]
    ferts = rec["fertilizers"]
    print(f"\n📌 Fertilizer Plan for {crop}:")
    for nutrient in ["N", "P", "K"]:
        key = f"{nutrient}_needed"
        fert_key = {"N":"Urea_kg_per_ha","P":"DAP_kg_per_ha","K":"MOP_kg_per_ha"}[nutrient]
        if deficits[key] > 0:
            print(f" • {nutrient} deficit: {deficits[key]} kg/ha → Apply {ferts[fert_key]} kg/ha fertilizer")
        elif deficits[key] < 0:
            print(f" • {nutrient} excess: {-deficits[key]} kg/ha → avoid additional {fert_key.split('_')[0]}")
    if deficits.get("pH_gap") is not None:
        if deficits["pH_gap"] > 0:
            print(f" • pH low by {deficits['pH_gap']} → apply lime")
        elif deficits["pH_gap"] < 0:
            print(f" • pH high by {-deficits['pH_gap']} → apply elemental sulfur")
    if deficits["temp_gap"] > 0:
        print(f" • Temperature low by {deficits['temp_gap']}°C → consider mulching/delay planting")
    elif deficits["temp_gap"] < 0:
        print(f" • Temperature high by {-deficits['temp_gap']}°C → shading/timing may help")
    if deficits["rain_gap"] > 0:
        print(f" • Rainfall deficit: {deficits['rain_gap']} mm → irrigation may be needed")
    elif deficits["rain_gap"] < 0:
        print(f" • Rainfall excess: {-deficits['rain_gap']} mm → ensure proper drainage")
    print("✅ Plan complete.\n")

# ---------------------------
# 4. Sensor reading helpers
# ---------------------------
def read_sensor_file(filename="sensor_readings.json"):
    if os.path.exists(filename):
        try:
            with open(filename,"r") as f:
                data = json.load(f)
            print(f"✅ Loaded readings from {filename}")
            return data
        except Exception as e:
            print("⚠️ Error reading file:", e)
    return None

def manual_sensor_entry():
    readings = {}
    for key in ["N","P","K","temperature","rainfall","pH"]:
        val = input(f"Enter {key} (leave blank to skip): ").strip()
        readings[key] = float(val) if val else 0
    return readings

# ---------------------------
# 5. ML predict
# ---------------------------
def ml_predict_crop(readings):
    if not use_ml: return None
    try:
        features = [[readings['N'], readings['P'], readings['K'], readings['temperature'], readings['rainfall']]]
        pred = model.predict(features)
        crop = le.inverse_transform(pred)[0] if le else str(pred[0])
        return crop
    except:
        return None

# ---------------------------
# 6. Interactive CLI
# ---------------------------
def run():
    while True:
        print("\n=== Crop Tool ===")
        print("Options:\n 1) Recommend crop from current soil\n 2) Show ideal soil for a crop\n 3) Crop + current soil → fertilizer plan\n 4) Quit")
        choice = input("Select option [1-4]: ").strip()

        if choice == "1":
            readings = read_sensor_file() or manual_sensor_entry()
            valid, msg = validate_crop_inputs(readings)
            print(msg)
            if not valid:
                print("⚠️ Using rule-based fallback")
                crop = rule_based_recommendation(readings.get("N",0), readings.get("P",0),
                                                 readings.get("K",0), readings.get("temperature",0),
                                                 readings.get("rainfall",0))
            else:
                crop = ml_predict_crop(readings) if use_ml else rule_based_recommendation(
                    readings.get("N",0), readings.get("P",0), readings.get("K",0),
                    readings.get("temperature",0), readings.get("rainfall",0)
                )
            print("✅ Recommended crop:", crop)

        elif choice == "2":
            crop = input("Enter crop name: ").strip().capitalize()
            if crop in ideal_soil:
                d = ideal_soil[crop]
                print(f"Ideal soil for {crop}: {d}")
            else:
                print("❌ No data for this crop.")

        elif choice == "3":
            crop = input("Enter crop name: ").strip().capitalize()
            readings = read_sensor_file() or manual_sensor_entry()
            rec = fertilizer_recommendation(crop, readings)
            print_fertilizer_plan(crop, rec)

        elif choice == "4":
            print("Goodbye.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    run()
