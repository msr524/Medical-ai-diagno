import streamlit as st
import requests
from PIL import Image
import io

# FastAPI backend endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Medical AI Diagnostic Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #4B5563; }
    .warning-box { background-color: #FEF3C7; padding: 15px; border-radius: 8px; border-left: 5px solid #F59E0B; color: #92400E; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏥 Medical AI Diagnostic Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning Classification & Explainable AI (Grad-CAM) with Gemini Report Generation</p>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
⚠️ <strong>DISCLAIMER:</strong> This tool is an AI-assisted draft generator for physician review only and does not constitute a formal medical diagnosis. All findings must be verified by a qualified medical professional.
</div>
""", unsafe_allow_html=True)

st.divider()

# Sidebar configuration and history
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Single Image Diagnostic", "Prediction History Log"])

if app_mode == "Single Image Diagnostic":
    st.subheader("Upload Medical Scan")
    uploaded_file = st.file_uploader("Upload Chest X-ray or Skin Lesion image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Scan")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Input Image", use_container_width=True)

        if st.button("Run Diagnostic Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing image, computing Grad-CAM, and drafting report..."):
                try:
                    # Reset pointer and prepare file for POST request
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code in [200, 201]:
                        data = response.json()
                        
                        with col2:
                            st.markdown("### Explainable AI (Grad-CAM)")
                            # Fetch Grad-CAM visualization from backend static mount safely
                            gradcam_path = data.get('gradcam_image', '')
                            gradcam_url = f"http://127.0.0.1:8000{gradcam_path}" if gradcam_path.startswith('/') else gradcam_path
                            st.image(gradcam_url, caption="Grad-CAM Heatmap (Model Focus Areas)", use_container_width=True)

                        st.success("Diagnostic analysis completed successfully!")
                        
                        # Display Results Metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Predicted Condition", data.get("predicted_class", "Unknown"))
                        m2.metric("Confidence Score", f"{data.get('confidence', 0.0)}%")

                        st.divider()

                        # Display Gemini AI Draft Report
                        st.markdown("### 📄 AI-Generated Physician Draft Report")
                        st.info(data.get("ai_report", "No report generated."))

                    else:
                        st.error(f"Server Error ({response.status_code}): {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the FastAPI backend. Ensure uvicorn is running on port 8000.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

elif app_mode == "Prediction History Log":
    st.subheader("Saved Prediction Records")
    try:
        res = requests.get("http://127.0.0.1:8000/predictions")
        if res.status_code == 200:
            records = res.json()
            if records:
                for rec in records:
                    conf_val = rec.get('confidence', 0.0)
                    # Handle confidence format whether stored as percentage or decimal
                    conf_display = conf_val if conf_val > 1 else conf_val * 100
                    with st.expander(f"ID: {rec.get('id')} | Class: {rec.get('predicted_class')} ({conf_display:.1f}%) | Time: {rec.get('timestamp')}"):
                        st.write(f"**Image Filename:** {rec.get('image_filename') or rec.get('filename')}")
                        st.text(rec.get('ai_report'))
            else:
                st.info("No prediction records found in the database yet.")
        else:
            st.error("Failed to fetch history logs from the server.")
    except Exception as e:
        st.error(f"Could not connect to backend server: {str(e)}")