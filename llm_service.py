import os
import time
import logging
from datetime import datetime
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError

# Configure logging for audit trails
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fetch API key strictly from environment variables
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def generate_medical_report(predicted_class: str, confidence: float, heatmap_description: str = "Central lung fields showing focal opacity concentration") -> str:
    fallback_report = (
        "AI-assisted report generation is temporarily unavailable (rate limit reached). "
        "Prediction and explainability results below are unaffected."
    )

    if not api_key:
        logger.warning(f"[{datetime.utcnow().isoformat()}] GEMINI_API_KEY environment variable is not set. Returning fallback report.")
        return fallback_report

    prompt = f"""
You are an expert clinical AI assistant. Generate a short, plain-language draft radiology report summary based on the following diagnostic metrics.

DIAGNOSTIC METRICS:
- Predicted Classification: {predicted_class}
- Model Confidence Score: {confidence * 100:.2f}%
- Grad-CAM Heatmap Localization: {heatmap_description}

STRICT INSTRUCTIONS:
1. Frame the output explicitly as an AI-assisted draft for physician review, never as a definitive medical diagnosis.
2. Keep the response concise, clear, and professional.
3. Include a reminder that clinical correlation and human review by a qualified physician are required.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] Failed to initialize Gemini model: {str(e)}")
        return fallback_report
    
    # Attempt 1: Primary generation call
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except ResourceExhausted as re:
        # Hard quota limit - do NOT retry, immediately return the required fallback string
        logger.error(f"[{datetime.utcnow().isoformat()}] GEMINI RATE LIMIT EXCEEDED (ResourceExhausted): {str(re)}")
        return fallback_report

    except GoogleAPICallError as g_err:
        # Transient network/API error - retry once after a short delay
        logger.warning(f"[{datetime.utcnow().isoformat()}] Transient Google API error encountered: {str(g_err)}. Retrying once...")
        time.sleep(2)
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as retry_err:
            logger.error(f"[{datetime.utcnow().isoformat()}] Retry failed for Google API call: {str(retry_err)}")
            return fallback_report

    except Exception as e:
        # Catch-all for any other unexpected execution errors
        logger.error(f"[{datetime.utcnow().isoformat()}] Unexpected error during Gemini report generation: {str(e)}")
        return fallback_report