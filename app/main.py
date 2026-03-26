from fastapi import FastAPI
from app.routes.predict_routes import router as predict_router
from app.models.detectron_food_model import get_model

app = FastAPI(title="Food ML Service")

app.include_router(predict_router)

@app.on_event("startup")
def warmup_model():
    # Carga anticipada para evitar que el primer request pague toda la inicialización
    get_model()

@app.get("/health")
async def health():
    return {"status": "ok"}