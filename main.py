# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# Crear app FastAPI
app = FastAPI()

# ====== Habilitar CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes reemplazar "*" por tu dominio Vercel al final
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Cargar modelo ======
with open("modelo_xgb_price.pkl", "rb") as f:
    model = pickle.load(f)


# ====== Modelo de request ======
class PredictRequest(BaseModel):
    features: list  # ejemplo: [30, 150, 2]


# ====== Endpoint de predicción ======
@app.post("/predict")
def predict(data: PredictRequest):
    X = np.array([data.features])
    pred = model.predict(X)
    return {"prediction": float(pred[0])}


# ====== Endpoint raíz ======
@app.get("/")
def home():
    return {"status": "ok", "message": "API funcionando"}
