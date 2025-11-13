from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔍 Iniciando servidor...")

print("📂 Archivos presentes en el contenedor:", os.listdir("."))

try:
    print("🔍 Intentando cargar modelo...")
    with open("modelo_xgb_price.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print("❌ ERROR cargando el modelo:", str(e))

class PredictRequest(BaseModel):
    features: list

@app.post("/predict")
def predict_price(data: PredictRequest):
    X = np.array([data.features])
    pred = model.predict(X)
    return {"prediction": float(pred[0])}

@app.get("/")
def home():
    return {"status": "ok", "message": "API funcionando"}
