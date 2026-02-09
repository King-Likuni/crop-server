# crop_app.py
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.menu import MDDropdownMenu
from kivy.metrics import dp
import joblib

# ---------------------------
# Load ML model (optional)
# ---------------------------
use_ml = False
model = None
le = None
try:
    model = joblib.load("crop_recommendation_model.pkl")
    le = joblib.load("label_encoder.pkl")
    use_ml = True
    print("✅ ML Model + label encoder loaded.")
except:
    print("⚠️ ML model not loaded, fallback to rules.")

# ---------------------------
# Ideal soil and rule-based fallback
# ---------------------------
ideal_soil = {
    "Maize":      {"N": 80, "P": 40, "K": 60, "temperature": 26, "rainfall": 120, "pH": 6.5},
    "Groundnuts": {"N": 25, "P": 30, "K": 30, "temperature": 28, "rainfall": 100, "pH": 6.0},
    "Cowpeas":    {"N": 20, "P": 25, "K": 20, "temperature": 30, "rainfall": 90,  "pH": 6.0},
    "Beans":      {"N": 40, "P": 50, "K": 30, "temperature": 22, "rainfall": 150, "pH": 6.0},
    "Watermelon": {"N": 60, "P": 55, "K": 40, "temperature": 32, "rainfall": 200, "pH": 6.5},
    "Mango":      {"N": 50, "P": 40, "K": 35, "temperature": 33, "rainfall": 70,  "pH": 6.0},
}

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

def ml_predict_crop(readings):
    if not use_ml:
        return None
    try:
        features = [[readings['N'], readings['P'], readings['K'], readings['temperature'], readings['rainfall']]]
        pred = model.predict(features)
        return le.inverse_transform(pred)[0] if le else str(pred[0])
    except:
        return None

# ---------------------------
# Fertilizer helpers
# ---------------------------
FERTILIZER_INFO = {
    "Urea": {"nutrient": "N", "pct_nutrient": 0.46},
    "DAP": {"nutrient": "P", "pct_nutrient": 0.18},
    "MOP": {"nutrient": "K", "pct_nutrient": 0.50},
}

def fertilizer_amounts_from_deficit(deficit_dict):
    rec = {}
    n_def = deficit_dict.get("N_needed", 0)
    p_def = deficit_dict.get("P_needed", 0)
    k_def = deficit_dict.get("K_needed", 0)

    rec["Urea_kg_per_ha"] = round(n_def / FERTILIZER_INFO["Urea"]["pct_nutrient"], 1) if n_def > 0 else 0
    rec["DAP_kg_per_ha"] = round(p_def / FERTILIZER_INFO["DAP"]["pct_nutrient"], 1) if p_def > 0 else 0
    rec["MOP_kg_per_ha"] = round(k_def / FERTILIZER_INFO["MOP"]["pct_nutrient"], 1) if k_def > 0 else 0

    return rec

def fertilizer_recommendation(crop, readings):
    crop = crop.capitalize()
    if crop not in ideal_soil:
        return None

    ideal = ideal_soil[crop]
    deficits = {
        "N_needed": round(ideal.get("N", 0) - readings.get("N", 0), 2),
        "P_needed": round(ideal.get("P", 0) - readings.get("P", 0), 2),
        "K_needed": round(ideal.get("K", 0) - readings.get("K", 0), 2),
        "temp_gap": round(ideal.get("temperature", 0) - readings.get("temperature", 0), 2),
        "rain_gap": round(ideal.get("rainfall", 0) - readings.get("rainfall", 0), 2),
        "pH_gap": round(ideal.get("pH", 0) - readings.get("pH", 0), 2) if readings.get("pH") else None
    }

    fertilizers = fertilizer_amounts_from_deficit(deficits)
    return {"deficits": deficits, "fertilizers": fertilizers}

# ---------------------------
# KV STRING WITH BORDERED RESULTS BOXES
# ---------------------------
KV = """
ScreenManager:
    MenuScreen:
    CropScreen:
    FertilizerScreen:

<MenuScreen>:
    name: "menu"
    MDBoxLayout:
        orientation: "vertical"
        padding: dp(20)
        spacing: dp(15)

        MDLabel:
            text: "🌾 Crop Recommendation App"
            halign: "center"
            font_style: "H4"

        MDRaisedButton:
            text: "1) Recommend Crop from Soil Inputs"
            pos_hint: {"center_x":0.5}
            on_release: app.root.current="crop"

        MDRaisedButton:
            text: "2) Show Ideal Soil for Crop"
            pos_hint: {"center_x":0.5}
            on_release: app.root.current="fertilizer"

        MDRaisedButton:
            text: "3) Fertilizer Plan for Crop & Current Soil"
            pos_hint: {"center_x":0.5}
            on_release: app.root.current="fertilizer"

        Widget:
        AnchorLayout:
            anchor_x: "right"
            anchor_y: "bottom"
            MDRaisedButton:
                text: "Quit"
                on_release: app.stop()

<CropScreen>:
    name: "crop"
    MDBoxLayout:
        orientation: "vertical"
        padding: dp(20)
        spacing: dp(10)

        MDLabel:
            text: "🌾 Crop Recommendation App"
            halign: "center"
            font_style: "H4"
            size_hint_y: None
            height: self.texture_size[1] + dp(20)

        ScrollView:
            size_hint_y: 0.48
            MDBoxLayout:
                orientation: "vertical"
                padding: dp(10)
                spacing: dp(10)
                size_hint_y: None
                height: self.minimum_height

                MDLabel:
                    text: "Enter Soil Readings"
                    halign: "center"
                    font_style: "H5"
                    size_hint_y: None
                    height: self.texture_size[1] + dp(15)

                MDTextField:
                    id: N_input
                    hint_text: "Nitrogen (N)"
                    input_filter: "float"
                    size_hint_y: None
                    height: dp(50)

                MDTextField:
                    id: P_input
                    hint_text: "Phosphorus (P)"
                    input_filter: "float"
                    size_hint_y: None
                    height: dp(50)

                MDTextField:
                    id: K_input
                    hint_text: "Potassium (K)"
                    input_filter: "float"
                    size_hint_y: None
                    height: dp(50)

                MDTextField:
                    id: temp_input
                    hint_text: "Temperature (°C)"
                    input_filter: "float"
                    size_hint_y: None
                    height: dp(50)

                MDTextField:
                    id: rain_input
                    hint_text: "Rainfall (mm)"
                    input_filter: "float"
                    size_hint_y: None
                    height: dp(50)

        # ----------- RESULTS BOX -----------
        MDBoxLayout:
            orientation: "vertical"
            size_hint_y: 0.52
            spacing: dp(10)

            MDBoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: dp(200)
                padding: dp(10)
                md_bg_color: 1, 1, 1, 1
                radius: [10, 10, 10, 10]
                md_border_color: 0, 0.5, 0, 1
                md_border_width: 2
                pos_hint: {"center_x": 0.5}
                size_hint_x: 0.95

                MDLabel:
                    text: "Results"
                    halign: "center"
                    font_style: "H6"
                    theme_text_color: "Custom"
                    text_color: 0, 0.5, 0, 1
                    size_hint_y: None
                    height: dp(30)

                ScrollView:
                    do_scroll_x: False
                    MDBoxLayout:
                        orientation: "vertical"
                        adaptive_height: True
                        padding: dp(10)

                        MDLabel:
                            id: result_label
                            text: ""
                            halign: "center"
                            valign: "top"
                            size_hint_y: None
                            height: self.texture_size[1] + dp(10)
                            text_size: self.width, None
                            theme_text_color: "Custom"
                            text_color: 0,0,0,1

            MDBoxLayout:
                orientation: "horizontal"
                size_hint_y: None
                height: dp(50)
                spacing: dp(10)

                MDRaisedButton:
                    text: "Show Recommended Crop"
                    size_hint_x: 0.5
                    on_release: root.recommend_crop()

                Widget:

                MDRaisedButton:
                    text: "Back to Menu"
                    size_hint_x: 0.5
                    on_release: app.root.current="menu"

<FertilizerScreen>:
    name: "fertilizer"
    MDBoxLayout:
        orientation: "vertical"
        padding: dp(20)
        spacing: dp(10)

        ScrollView:
            size_hint_y: 0.48
            MDBoxLayout:
                orientation: "vertical"
                padding: dp(10)
                spacing: dp(10)
                size_hint_y: None
                height: self.minimum_height

                MDLabel:
                    text: "Crop & Current Soil"
                    halign: "center"
                    font_style: "H5"
                    size_hint_y: None
                    height: self.texture_size[1] + dp(15)

                MDLabel:
                    text: "Select Crop"
                    halign: "center"

                MDRaisedButton:
                    id: crop_dropdown_btn
                    text: "Choose Crop"
                    size_hint_x: 0.8
                    pos_hint: {"center_x": 0.5}
                    on_release: root.open_crop_menu()

                MDTextField:
                    id: N_input
                    hint_text: "Nitrogen (N)"
                    input_filter: "float"

                MDTextField:
                    id: P_input
                    hint_text: "Phosphorus (P)"
                    input_filter: "float"

                MDTextField:
                    id: K_input
                    hint_text: "Potassium (K)"
                    input_filter: "float"

                MDTextField:
                    id: temp_input
                    hint_text: "Temperature (°C)"
                    input_filter: "float"

                MDTextField:
                    id: rain_input
                    hint_text: "Rainfall (mm)"
                    input_filter: "float"

                MDTextField:
                    id: pH_input
                    hint_text: "pH (optional)"
                    input_filter: "float"

        # ----------- RESULTS BOX -----------
        MDBoxLayout:
            orientation: "vertical"
            size_hint_y: 0.52
            spacing: dp(10)

            MDBoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: dp(230)
                padding: dp(10)
                md_bg_color: 1, 1, 1, 1
                radius: [10, 10, 10, 10]
                md_border_color: 0, 0.5, 0, 1
                md_border_width: 2
                pos_hint: {"center_x": 0.5}
                size_hint_x: 0.95

                MDLabel:
                    text: "Results"
                    halign: "center"
                    font_style: "H6"
                    theme_text_color: "Custom"
                    text_color: 0, 0.5, 0, 1
                    size_hint_y: None
                    height: dp(30)

                ScrollView:
                    do_scroll_x: False
                    MDBoxLayout:
                        orientation: "vertical"
                        adaptive_height: True
                        padding: dp(10)

                        MDLabel:
                            id: plan_label
                            text: ""
                            halign: "center"
                            valign: "top"
                            size_hint_y: None
                            height: self.texture_size[1] + dp(10)
                            text_size: self.width, None
                            theme_text_color: "Custom"
                            text_color: 0,0,0,1

            MDBoxLayout:
                orientation: "horizontal"
                size_hint_y: None
                height: dp(50)
                spacing: dp(10)

                MDRaisedButton:
                    text: "Show Ideal Soil"
                    size_hint_x: 0.33
                    on_release: root.show_ideal_soil()

                MDRaisedButton:
                    text: "Generate Fertilizer Plan"
                    size_hint_x: 0.33
                    on_release: root.generate_plan()

                MDRaisedButton:
                    text: "Back to Menu"
                    size_hint_x: 0.33
                    on_release: app.root.current="menu"
"""

# ---------------------------
# Screen classes
# ---------------------------
class MenuScreen(Screen):
    pass

class CropScreen(Screen):
    def recommend_crop(self):
        try:
            readings = {
                'N': float(self.ids.N_input.text),
                'P': float(self.ids.P_input.text),
                'K': float(self.ids.K_input.text),
                'temperature': float(self.ids.temp_input.text),
                'rainfall': float(self.ids.rain_input.text),
            }
        except:
            self.ids.result_label.text = "❌ Please enter valid numbers."
            return

        crop = ml_predict_crop(readings) if use_ml else rule_based_recommendation(**readings)
        if not crop:
            crop = rule_based_recommendation(**readings)

        self.ids.result_label.text = f"✅ Recommended Crop: {crop}"

class FertilizerScreen(Screen):
    def on_pre_enter(self):
        self.menu_items = [
            {
                "text": crop,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=crop: self.select_crop(x)
            } for crop in ideal_soil.keys()
        ]

        self.crop_menu = MDDropdownMenu(
            caller=self.ids.crop_dropdown_btn,
            items=self.menu_items,
            width_mult=4,
        )

    def open_crop_menu(self):
        self.crop_menu.open()

    def select_crop(self, crop_name):
        self.ids.crop_dropdown_btn.text = crop_name
        self.crop_menu.dismiss()

    def generate_plan(self):
        try:
            crop = self.ids.crop_dropdown_btn.text
            if crop not in ideal_soil:
                self.ids.plan_label.text = "❌ Please select a valid crop."
                return

            readings = {
                'N': float(self.ids.N_input.text),
                'P': float(self.ids.P_input.text),
                'K': float(self.ids.K_input.text),
                'temperature': float(self.ids.temp_input.text),
                'rainfall': float(self.ids.rain_input.text),
                'pH': float(self.ids.pH_input.text) if self.ids.pH_input.text else None,
            }
        except:
            self.ids.plan_label.text = "❌ Enter valid numbers."
            return

        rec = fertilizer_recommendation(crop, readings)
        if not rec:
            self.ids.plan_label.text = "❌ No ideal data for this crop."
            return

        deficits = rec["deficits"]
        ferts = rec["fertilizers"]

        text = f"Fertilizer Plan for {crop}:\n"
        for nut in ["N", "P", "K"]:
            need_key = f"{nut}_needed"
            fert_key = {
                "N": "Urea_kg_per_ha",
                "P": "DAP_kg_per_ha",
                "K": "MOP_kg_per_ha",
            }[nut]

            if deficits[need_key] > 0:
                text += f"{nut} deficit: {deficits[need_key]} → Apply {ferts[fert_key]} kg/ha\n"
            else:
                text += f"{nut} OK (No deficit)\n"

        self.ids.plan_label.text = text

    def show_ideal_soil(self):
        crop = self.ids.crop_dropdown_btn.text
        if crop not in ideal_soil:
            self.ids.plan_label.text = "❌ Please select a valid crop."
            return

        ideal = ideal_soil[crop]

        text = f"Ideal Soil for {crop}:\n"
        text += f"N: {ideal['N']}, P: {ideal['P']}, K: {ideal['K']}\n"
        text += f"Temperature: {ideal['temperature']}°C, Rainfall: {ideal['rainfall']}mm\n"
        text += f"pH: {ideal.get('pH', 'N/A')}"

        self.ids.plan_label.text = text

# ---------------------------
# App class
# ---------------------------
class CropApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Green"
        return Builder.load_string(KV)

if __name__ == "__main__":
    CropApp().run()
