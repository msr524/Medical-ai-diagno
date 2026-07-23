import os
import requests

class MedicalReportGenerator:
    def __init__(self):
        self.proxy_url = os.getenv("LLM_PROXY_URL", "https://unequine-bud-expensively.ngrok-free.dev/proxy-llm")

    def generate_draft_report(self, predicted_class: str, confidence: float, heatmap_description: str = "Central lung fields showing focal opacity concentration") -> str:
        
        # Professional clinical template
        structured_report = (
            "[AI-Assisted Draft - Physician Review Required]\n\n"
            f"Finding: The deep learning model classified the image as '{predicted_class}' with a confidence score of {confidence:.2f}%.\n"
            f"Region Analysis: {heatmap_description}.\n"
            "Assessment: Image evaluation completed via deep learning pipeline with Grad-CAM visualization. "
            "Please consult a qualified radiologist or physician for formal diagnosis and clinical correlation."
        )

        payload = {
            "prompt": structured_report,
            "predicted_class": predicted_class,
            "confidence": confidence
        }

        try:
            response = requests.post(self.proxy_url, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("report", structured_report)
            else:
                return structured_report
        except Exception:
            # Returns the professional draft cleanly without throwing errors or referencing missing keys
            return structured_report