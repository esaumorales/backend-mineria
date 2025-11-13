from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# ========== CORS FIX PARA RAILWAY ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Permite todo
    allow_origin_regex=".*",         # <--- FIX IMPORTANTE PARA RAILWAY
    allow_credentials=True,
    allow_methods=["*"],             # Permite OPTIONS, GET, POST, etc.
    allow_headers=["*"],             # Permite Authorization, Content-Type, etc.
    expose_headers=["*"],            # Expone headers en respuesta
)

# ========== Cargar modelo ==========
with open("modelo_xgb_price.pkl", "rb") as f:
    model = pickle.load(f)

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
