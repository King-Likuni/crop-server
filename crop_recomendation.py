from kivy.lang import Builder
from kivymd.app import MDApp

KV = """
ScreenManager:
    MenuScreen:
        name: "menu"
    CropScreen:
        name: "crop"
    FertilizerScreen:
        name: "fert"

<MenuScreen@MDScreen>:
    MDFloatLayout:
        MDTopAppBar:
            title: "SmartAgri Dashboard"
            pos_hint: {"top": 1}
            elevation: 4

        MDLabel:
            text: "Select a Service"
            halign: "center"
            pos_hint: {"center_y": 0.85}
            theme_text_color: "Primary"

        MDCard:
            size_hint: 0.85, 0.22
            pos_hint: {"center_x": 0.5, "center_y": 0.6}
            radius: [25]
            elevation: 3
            orientation: "vertical"
            padding: 10

            MDLabel:
                text: "Crop Recommendation"
                halign: "center"
                theme_text_color: "Primary"

            MDRaisedButton:
                text: "Open"
                pos_hint: {"center_x": 0.5}
                on_release: app.root.current = "crop"

        MDCard:
            size_hint: 0.85, 0.22
            pos_hint: {"center_x": 0.5, "center_y": 0.35}
            radius: [25]
            elevation: 3
            orientation: "vertical"
            padding: 10

            MDLabel:
                text: "Fertilizer Recommendation"
                halign: "center"
                theme_text_color: "Primary"

            MDRaisedButton:
                text: "Open"
                pos_hint: {"center_x": 0.5}
                on_release: app.root.current = "fert"

        MDRaisedButton:
            text: "Quit"
            pos_hint: {"center_x": 0.5, "y": 0.05}
            on_release: app.stop()


<CropScreen@MDScreen>:
    MDFloatLayout:
        MDTopAppBar:
            title: "Crop Recommendation"
            pos_hint: {"top": 1}
            left_action_items: [["arrow-left", lambda x: app.go_back()]]

        MDBoxLayout:
            orientation: "horizontal"
            size_hint_x: 0.8
            pos_hint: {"center_x": 0.5, "center_y": 0.75}
            spacing: 10

            MDLabel:
                text: "Nitrogen (N):"
                size_hint_x: 0.4
                halign: "right"

            MDTextField:
                id: nitrogen
                hint_text: "Enter value"
                size_hint_x: 0.6

        MDBoxLayout:
            orientation: "horizontal"
            size_hint_x: 0.8
            pos_hint: {"center_x": 0.5, "center_y": 0.63}
            spacing: 10

            MDLabel:
                text: "Phosphorus (P):"
                size_hint_x: 0.4
                halign: "right"

            MDTextField:
                id: phosphorus
                hint_text: "Enter value"
                size_hint_x: 0.6

        MDBoxLayout:
            orientation: "horizontal"
            size_hint_x: 0.8
            pos_hint: {"center_x": 0.5, "center_y": 0.51}
            spacing: 10

            MDLabel:
                text: "Potassium (K):"
                size_hint_x: 0.4
                halign: "right"

            MDTextField:
                id: potassium
                hint_text: "Enter value"
                size_hint_x: 0.6

        MDRaisedButton:
            text: "Recommend Crop"
            pos_hint: {"center_x": 0.5, "center_y": 0.38}
            on_release: result_label.text = f"Recommended Crop: Maize"

        MDLabel:
            id: result_label
            halign: "center"
            pos_hint: {"center_y": 0.25}
            theme_text_color: "Primary"

        MDRaisedButton:
            text: "Back"
            pos_hint: {"x": 0.05, "y": 0.05}
            size_hint_x: 0.2
            on_release: app.go_back()


<FertilizerScreen@MDScreen>:
    MDFloatLayout:
        MDTopAppBar:
            title: "Fertilizer Recommendation"
            pos_hint: {"top": 1}
            left_action_items: [["arrow-left", lambda x: app.go_back()]]

        MDBoxLayout:
            orientation: "horizontal"
            size_hint_x: 0.8
            pos_hint: {"center_x": 0.5, "center_y": 0.7}
            spacing: 10

            MDLabel:
                text: "Crop Type:"
                size_hint_x: 0.4
                halign: "right"

            MDTextField:
                id: crop_type
                hint_text: "Enter crop"
                size_hint_x: 0.6

        MDBoxLayout:
            orientation: "horizontal"
            size_hint_x: 0.8
            pos_hint: {"center_x": 0.5, "center_y": 0.58}
            spacing: 10

            MDLabel:
                text: "Soil Type:"
                size_hint_x: 0.4
                halign: "right"

            MDTextField:
                id: soil_type
                hint_text: "Enter soil"
                size_hint_x: 0.6

        MDRaisedButton:
            text: "Get Fertilizer Mix"
            pos_hint: {"center_x": 0.5, "center_y": 0.44}
            on_release: fert_result.text = f"Recommended NPK: 10-20-10"

        MDLabel:
            id: fert_result
            halign: "center"
            pos_hint: {"center_y": 0.3}
            theme_text_color: "Primary"

        MDRaisedButton:
            text: "Back"
            pos_hint: {"x": 0.05, "y": 0.05}
            size_hint_x: 0.2
            on_release: app.go_back()
"""

class SmartAgri(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.theme_style = "Light"
        return Builder.load_string(KV)

    def go_back(self):
        self.root.current = "menu"


if __name__ == "__main__":
    SmartAgri().run()
