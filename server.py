from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import traceback
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For server
import matplotlib.pyplot as plt
import io
import base64
from fpdf import FPDF
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# =========================================================
# CONFIGURATION
# =========================================================
FEATURE_NAMES = ["N", "P", "K", "moisture", "temperature", "pH"]
Z_THRESHOLD = 3.0
CONFIDENCE_THRESHOLD = 0.60

PHYSICAL_LIMITS = {
    "N": (0, 200), "P": (0, 200), "K": (0, 300),
    "moisture": (0, 100), "temperature": (-10, 60), "pH": (3.0, 10.0)
}

# =========================================================
# IDEAL RANGES FOR ALL CROPS
# =========================================================
IDEAL_RANGES = {
    "maize": {
        "N": {"min": 60, "max": 100},
        "P": {"min": 35, "max": 60},
        "K": {"min": 15, "max": 25},
        "pH": {"min": 5.5, "max": 7.0},
        "moisture": {"min": 55, "max": 75},
        "temperature": {"min": 18, "max": 26}
    },
    "beans": {
        "N": {"min": 0, "max": 40},
        "P": {"min": 55, "max": 80},
        "K": {"min": 15, "max": 25},
        "pH": {"min": 5.5, "max": 6.0},
        "moisture": {"min": 18, "max": 25},
        "temperature": {"min": 15, "max": 25}
    },
    "cowpeas": {
        "N": {"min": 0, "max": 40},
        "P": {"min": 35, "max": 60},
        "K": {"min": 15, "max": 25},
        "pH": {"min": 6.2, "max": 7.2},
        "moisture": {"min": 80, "max": 90},
        "temperature": {"min": 27, "max": 30}
    },
    "groundnuts": {
        "N": {"min": 0, "max": 40},
        "P": {"min": 35, "max": 60},
        "K": {"min": 15, "max": 25},
        "pH": {"min": 3.5, "max": 9.9},
        "moisture": {"min": 40, "max": 65},
        "temperature": {"min": 24, "max": 32}
    },
    "mango": {
        "N": {"min": 0, "max": 40},
        "P": {"min": 15, "max": 40},
        "K": {"min": 25, "max": 35},
        "pH": {"min": 4.5, "max": 7.0},
        "moisture": {"min": 45, "max": 55},
        "temperature": {"min": 27, "max": 36}
    },
    "watermelon": {
        "N": {"min": 80, "max": 120},
        "P": {"min": 5, "max": 30},
        "K": {"min": 45, "max": 55},
        "pH": {"min": 6.0, "max": 7.0},
        "moisture": {"min": 80, "max": 90},
        "temperature": {"min": 24, "max": 27}
    }
}

# =========================================================
# FERTILIZER RECOMMENDATIONS
# =========================================================
FERTILIZER_ADVICE = {
    "N_low": "🔴 Nitrogen is low. Add {:.1f} kg/ha Urea (46% N) or use organic manure.",
    "N_high": "🟡 Nitrogen is high. Reduce nitrogen fertilizers and plant nitrogen-fixing cover crops.",
    "N_optimal": "🟢 Nitrogen level is optimal. Maintain current practices.",
    
    "P_low": "🔴 Phosphorus is low. Add {:.1f} kg/ha SSP (20% P2O5) or DAP.",
    "P_high": "🟡 Phosphorus is high. Avoid phosphate fertilizers for next season.",
    "P_optimal": "🟢 Phosphorus level is optimal. Maintain current practices.",
    
    "K_low": "🔴 Potassium is low. Add {:.1f} kg/ha MOP (60% K2O) or organic compost.",
    "K_high": "🟡 Potassium is high. Reduce potash application.",
    "K_optimal": "🟢 Potassium level is optimal. Maintain current practices.",
    
    "pH_low": "🔴 Soil is too acidic. Add {:.1f} tons/ha lime to raise pH.",
    "pH_high": "🔴 Soil is too alkaline. Add sulfur or organic matter to lower pH.",
    "pH_optimal": "🟢 pH level is optimal. No action needed.",
    
    "moisture_low": "🔴 Moisture is low. Increase irrigation frequency.",
    "moisture_high": "🟡 Moisture is high. Improve drainage and reduce irrigation.",
    "moisture_optimal": "🟢 Moisture level is optimal. Maintain current irrigation."
}

# =========================================================
# LOAD MODEL + ENCODER (your existing code)
# =========================================================
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
except Exception as e:
    print("⚠️ Model loading failed:", e)

# =========================================================
# LOAD FEATURE STATS (your existing code)
# =========================================================
feature_means = None
feature_stds = None

try:
    if os.path.exists("feature_means.pkl") and os.path.exists("feature_stds.pkl"):
        feature_means = joblib.load("feature_means.pkl")
        feature_stds = joblib.load("feature_stds.pkl")
        print("✅ Feature statistics loaded")
    else:
        print("❌ Feature stats not found")
except Exception as e:
    print("⚠️ Failed to load feature stats:", e)

# =========================================================
# MEMORY STORAGE
# =========================================================
latest_sensor_data = {}
latest_recommendation = None
latest_confidence = None

# =========================================================
# VALIDATION FUNCTIONS (your existing code)
# =========================================================
def within_physical_limits(sensor_values):
    for key, (min_val, max_val) in PHYSICAL_LIMITS.items():
        if not (min_val <= sensor_values[key] <= max_val):
            return False
    return True

def within_zscore(sensor_values):
    if feature_means is None or feature_stds is None:
        return False
    try:
        sensor_array = np.array([sensor_values[f] for f in FEATURE_NAMES])
        z_scores = np.abs((sensor_array - feature_means) / feature_stds)
        return np.all(z_scores <= Z_THRESHOLD)
    except:
        return False

# =========================================================
# NEW ENDPOINT 1: GET IDEAL RANGES FOR A CROP
# =========================================================
@app.route("/ideal-ranges/<crop>", methods=["GET"])
def get_ideal_ranges(crop):
    """Get ideal soil ranges for a specific crop"""
    crop = crop.lower()
    if crop in IDEAL_RANGES:
        return jsonify({
            "status": "success",
            "crop": crop,
            "ranges": IDEAL_RANGES[crop]
        })
    else:
        # Return maize as default
        return jsonify({
            "status": "success",
            "crop": "maize (default)",
            "ranges": IDEAL_RANGES["maize"]
        })

# =========================================================
# NEW ENDPOINT 2: GET ALL CROPS
# =========================================================
@app.route("/crops", methods=["GET"])
def get_crops():
    """Get list of all supported crops"""
    return jsonify({
        "status": "success",
        "crops": list(IDEAL_RANGES.keys())
    })

# =========================================================
# NEW ENDPOINT 3: GENERATE FERTILIZER PLAN
# =========================================================
@app.route("/fertilizer-plan", methods=["POST"])
def fertilizer_plan():
    """Generate fertilizer plan based on sensor data and crop"""
    try:
        data = request.get_json(force=True)
        crop = data.get("crop", "maize").lower()
        sensor = data.get("sensor_data", latest_sensor_data)
        
        if not sensor:
            return jsonify({"status": "error", "message": "No sensor data provided"}), 400
        
        # Get ideal ranges for the crop
        ranges = IDEAL_RANGES.get(crop, IDEAL_RANGES["maize"])
        
        plan = []
        
        # Check NPK
        for nutrient in ["N", "P", "K"]:
            current = sensor.get(nutrient, 0)
            ideal_min = ranges[nutrient]["min"]
            ideal_max = ranges[nutrient]["max"]
            
            if current < ideal_min:
                deficit = ideal_min - current
                plan.append(FERTILIZER_ADVICE[f"{nutrient}_low"].format(deficit))
            elif current > ideal_max:
                excess = current - ideal_max
                plan.append(FERTILIZER_ADVICE[f"{nutrient}_high"])
            else:
                plan.append(FERTILIZER_ADVICE[f"{nutrient}_optimal"])
        
        # Check pH
        current_ph = sensor.get("pH", 6.5)
        ph_min = ranges["pH"]["min"]
        ph_max = ranges["pH"]["max"]
        
        if current_ph < ph_min:
            diff = ph_min - current_ph
            plan.append(FERTILIZER_ADVICE["pH_low"].format(diff))
        elif current_ph > ph_max:
            diff = current_ph - ph_max
            plan.append(FERTILIZER_ADVICE["pH_high"].format(diff))
        else:
            plan.append(FERTILIZER_ADVICE["pH_optimal"])
        
        # Check moisture
        current_moisture = sensor.get("moisture", 50)
        moisture_min = ranges["moisture"]["min"]
        moisture_max = ranges["moisture"]["max"]
        
        if current_moisture < moisture_min:
            plan.append(FERTILIZER_ADVICE["moisture_low"])
        elif current_moisture > moisture_max:
            plan.append(FERTILIZER_ADVICE["moisture_high"])
        else:
            plan.append(FERTILIZER_ADVICE["moisture_optimal"])
        
        return jsonify({
            "status": "success",
            "crop": crop,
            "plan": plan
        })
        
    except Exception as e:
        print("❌ Error generating fertilizer plan:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================================
# NEW ENDPOINT 4: GENERATE NPK CHART
# =========================================================
@app.route("/chart/npk", methods=["POST"])
def generate_npk_chart():
    """Generate NPK distribution chart"""
    try:
        data = request.get_json(force=True)
        sensor = data.get("sensor_data", latest_sensor_data)
        crop = data.get("crop", "maize")
        
        if not sensor:
            return jsonify({"status": "error", "message": "No sensor data"}), 400
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Current NPK values
        values = [sensor.get("N", 0), sensor.get("P", 0), sensor.get("K", 0)]
        nutrients = ["N", "P", "K"]
        colors = ["#4CAF50", "#FFC107", "#2196F3"]
        
        # Pie chart
        ax1.pie(values, labels=nutrients, autopct="%1.1f%%",
                startangle=90, colors=colors)
        ax1.set_title("Current NPK Distribution")
        
        # Bar chart with ranges
        ranges = IDEAL_RANGES.get(crop, IDEAL_RANGES["maize"])
        x_pos = np.arange(len(nutrients))
        width = 0.35
        
        # Plot current values
        ax2.bar(x_pos, values, width, color='orange', label='Current')
        
        # Plot ideal ranges
        for i, nutrient in enumerate(nutrients):
            min_val = ranges[nutrient]["min"]
            max_val = ranges[nutrient]["max"]
            ax2.bar(i + width, max_val - min_val, width, 
                   bottom=min_val, color='green', alpha=0.3, label='Ideal Range' if i == 0 else "")
        
        ax2.set_xticks(x_pos + width/2)
        ax2.set_xticklabels(nutrients)
        ax2.set_ylabel("Value")
        ax2.set_title(f"Ideal Ranges for {crop.title()}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return jsonify({
            "status": "success",
            "image": img_base64
        })
        
    except Exception as e:
        print("❌ Error generating chart:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================================
# NEW ENDPOINT 5: GENERATE SOIL PARAMETERS CHART
# =========================================================
@app.route("/chart/soil", methods=["POST"])
def generate_soil_chart():
    """Generate moisture, pH, temperature chart"""
    try:
        data = request.get_json(force=True)
        sensor = data.get("sensor_data", latest_sensor_data)
        crop = data.get("crop", "maize")
        
        if not sensor:
            return jsonify({"status": "error", "message": "No sensor data"}), 400
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        categories = ['moisture', 'pH', 'temperature']
        current_values = [sensor.get(c, 0) for c in categories]
        x_pos = np.arange(len(categories))
        width = 0.35
        
        # Plot current values
        bars = ax.bar(x_pos, current_values, width, color='#FF5722', label='Current')
        
        # Plot ideal ranges
        ranges = IDEAL_RANGES.get(crop, IDEAL_RANGES["maize"])
        for i, cat in enumerate(categories):
            if cat == 'pH':
                key = 'pH'
            else:
                key = cat
            min_val = ranges[key]["min"]
            max_val = ranges[key]["max"]
            
            # Draw range bar
            ax.bar(i + width, max_val - min_val, width, 
                   bottom=min_val, color='green', alpha=0.3, label='Ideal Range' if i == 0 else "")
            
            # Add range markers
            ax.plot([i + width/2, i + width/2], [min_val, max_val], 
                   color='darkgreen', linewidth=2)
        
        ax.set_xticks(x_pos + width/2)
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylabel("Value")
        ax.set_title(f"Soil Parameters vs Ideal for {crop.title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return jsonify({
            "status": "success",
            "image": img_base64
        })
        
    except Exception as e:
        print("❌ Error generating soil chart:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================================
# NEW ENDPOINT 6: GENERATE PDF REPORT
# =========================================================
@app.route("/report/pdf", methods=["POST"])
def generate_pdf_report():
    """Generate PDF report with all data"""
    try:
        data = request.get_json(force=True)
        sensor = data.get("sensor_data", latest_sensor_data)
        crop = data.get("crop", latest_recommendation or "maize")
        
        if not sensor:
            return jsonify({"status": "error", "message": "No sensor data"}), 400
        
        # Get fertilizer plan
        fert_response = fertilizer_plan()
        fert_data = fert_response.get_json() if hasattr(fert_response, 'get_json') else {"plan": []}
        
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Crop Recommendation Report", ln=True, align='C')
        pdf.ln(10)
        
        # Date
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)
        
        # Sensor Data
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Sensor Data:", ln=True)
        pdf.set_font("Arial", '', 11)
        for key, value in sensor.items():
            pdf.cell(0, 8, f"  {key}: {value:.2f}", ln=True)
        pdf.ln(5)
        
        # Recommended Crop
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Recommended Crop: {crop.title()}", ln=True)
        pdf.ln(5)
        
        # Fertilizer Plan
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Fertilizer Plan:", ln=True)
        pdf.set_font("Arial", '', 11)
        for item in fert_data.get("plan", []):
            # Handle UTF-8 emojis
            clean_item = item.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 8, clean_item)
        
        # Output
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        
        return send_file(
            pdf_output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"crop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
    except Exception as e:
        print("❌ Error generating PDF:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================================
# NEW ENDPOINT 7: GET COMPLETE DASHBOARD DATA
# =========================================================
@app.route("/dashboard", methods=["GET"])
def get_dashboard():
    """Get all data for dashboard in one request"""
    crop = latest_recommendation or "maize"
    
    return jsonify({
        "status": "success",
        "sensor_data": latest_sensor_data,
        "recommended_crop": crop,
        "confidence": latest_confidence,
        "ideal_ranges": IDEAL_RANGES.get(crop, IDEAL_RANGES["maize"]),
        "model_loaded": model_loaded
    })

# =========================================================
# YOUR EXISTING ENDPOINTS (keep them)
# =========================================================

@app.route("/sensor-data", methods=["POST"])
def sensor_data():
    global latest_sensor_data, latest_recommendation, latest_confidence
    latest_confidence = None

    try:
        data = request.get_json(force=True)

        # Validate required keys
        for key in FEATURE_NAMES:
            if key not in data:
                return jsonify({"status": "error", "message": f"Missing key: {key}"}), 400

        # Convert to float
        latest_sensor_data = {key: float(data[key]) for key in FEATURE_NAMES}
        print("📡 Sensor data:", latest_sensor_data)

        # -------------------------------------------------
        # 1️⃣ Physical validation
        # -------------------------------------------------
        if not within_physical_limits(latest_sensor_data):
            latest_recommendation = "No crop recommended (physically impossible values)"

        # -------------------------------------------------
        # 2️⃣ Z-score validation
        # -------------------------------------------------
        elif not within_zscore(latest_sensor_data):
            latest_recommendation = "No crop recommended "

        # -------------------------------------------------
        # 3️⃣ ML Prediction + confidence check
        # -------------------------------------------------
        elif model_loaded:
            features = [latest_sensor_data[f] for f in FEATURE_NAMES]
            prediction_index = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            confidence = float(np.max(probabilities))
            latest_confidence = round(confidence, 2)

            if confidence < CONFIDENCE_THRESHOLD:
                latest_recommendation = "No crop recommended"
            else:
                latest_recommendation = le.inverse_transform([prediction_index])[0]

        else:
            latest_recommendation = "Model unavailable"

        return jsonify({
            "status": "success",
            "sensor_data": latest_sensor_data,
            "recommended_crop": latest_recommendation,
            "confidence": latest_confidence,
            "model_loaded": model_loaded
        })

    except Exception as e:
        print("❌ Error:", e)
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/recommend-crops", methods=["GET"])
def recommend_crops():
    if not latest_sensor_data:
        return jsonify({"status": "error", "message": "No sensor data received yet"}), 404

    return jsonify({
        "status": "success",
        "recommended_crop": latest_recommendation,
        "confidence": latest_confidence,
        "sensor_data": latest_sensor_data
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "feature_stats_loaded": feature_means is not None,
        "latest_recommendation": latest_recommendation,
        "latest_confidence": latest_confidence,
        "available_endpoints": [
            "/ (GET) - Health check",
            "/crops (GET) - List all crops",
            "/ideal-ranges/<crop> (GET) - Get ideal ranges",
            "/sensor-data (POST) - Submit sensor data",
            "/recommend-crops (GET) - Get recommendation",
            "/fertilizer-plan (POST) - Get fertilizer plan",
            "/chart/npk (POST) - Get NPK chart",
            "/chart/soil (POST) - Get soil parameters chart",
            "/report/pdf (POST) - Download PDF report",
            "/dashboard (GET) - Get all data"
        ]
    })

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


