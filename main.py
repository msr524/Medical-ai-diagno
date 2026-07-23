import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from src.database import engine, Base, get_db
from src.models import PredictionRecord
from src.ml_pipeline import predict_and_generate_gradcam
from src.llm_service import generate_medical_report

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Medical AI Diagnostic API",
    description="Production-grade AI diagnostic system with Grad-CAM explainability and Gemini reporting.",
    version="1.0.0"
)

# Directories for uploads and outputs
UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static folders so saved images can be served if needed
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Medical AI Diagnostic System API is running successfully."}

@app.post("/predict", status_code=status.HTTP_201_CREATED)
async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint to accept a medical image, run inference, generate Grad-CAM, 
    compile an LLM report, and persist everything to the SQLite database.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload an image (.png, .jpg, .jpeg, .bmp, .webp)."
        )

    try:
        # Generate unique filenames to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        file_ext = os.path.splitext(file.filename)[1]
        input_filename = f"input_{unique_id}{file_ext}"
        gradcam_filename = f"gradcam_{unique_id}.png"
        
        input_path = os.path.join(UPLOAD_DIR, input_filename)
        gradcam_path = os.path.join(OUTPUT_DIR, gradcam_filename)

        # Save uploaded file locally
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ML inference and Grad-CAM generation
        pred_class, confidence = predict_and_generate_gradcam(
            image_path=input_path,
            output_gradcam_path=gradcam_path
        )

        # Generate Gemini LLM report draft
        ai_report = generate_medical_report(pred_class, confidence)

        # Persist record in database
        db_record = PredictionRecord(
            image_filename=input_filename,
            predicted_class=pred_class,
            confidence=confidence,
            gradcam_image_path=gradcam_path,
            ai_report=ai_report
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return {
            "prediction_id": db_record.id,
            "filename": input_filename,
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 2),
            "gradcam_image": f"/outputs/{gradcam_filename}",
            "ai_report": ai_report,
            "timestamp": db_record.timestamp.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during diagnostic processing: {str(e)}"
        )

@app.get("/predictions", status_code=status.HTTP_200_OK)
def get_prediction_history(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    """Retrieves prediction history logs from the database."""
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).offset(skip).limit(limit).all()
        return records
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch prediction history: {str(e)}"
        )