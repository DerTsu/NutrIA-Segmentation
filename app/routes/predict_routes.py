import io
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.core.settings import MAX_IMAGE_BYTES
from app.models.detectron_food_model import get_model
from app.schemas.predict_schema import PredictionResponse

router = APIRouter(prefix="/predict", tags=["Predict"])

@router.post("", response_model=PredictionResponse)
async def predict_food(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el tamaño permitido.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="La imagen es inválida o está corrupta.")

    model = get_model()
    return model.predict(image_bytes)