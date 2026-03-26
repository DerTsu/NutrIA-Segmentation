from pydantic import BaseModel
from typing import List, Optional

class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    score: float
    area_pct: Optional[float] = None

class PredictionResponse(BaseModel):
    detections: List[DetectionItem]
    summary_text: str