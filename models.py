import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from src.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_filename = Column(String(255), nullable=False)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    gradcam_image_path = Column(String(255), nullable=True)
    ai_report = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)