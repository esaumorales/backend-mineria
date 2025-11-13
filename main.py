# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Crear app primero
app = FastAPI()

# ========== CORS (DEBE IR AQUÍ ANTES DE CUALQUIER OTRA COSA) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Cámbialo después por tu dominio Vercel
    allow_credentials=True,
    allow_methods=["*"],   # IMPORTANTE PARA OPTIONS
    allow_headers=["*"],   # IMPORTANTE PARA OPTIONS
)

# ========== Cargar modelo ==========
with open("modelo_xgb_price.pkl", "rb") as f:
    model = pickle.load(f)

# ========== Request Model ==========
class PredictRequest(BaseModel):
    features: list

# ========== Endpoint POST ==========
@app.post("/predict")
def predict_price(data: PredictRequest):
    X = np.array([data.features])
    pred = model.predict(X)
    return {"prediction": float(pred[0])}

# ========== Endpoint GET raíz ==========
@app.get("/")
def home():
    return {"status": "ok", "message": "API funcionando"}
