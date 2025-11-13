# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# ===== Cargar modelo =====
with open("modelo_xgb_price.pkl", "rb") as f:
    model = pickle.load(f)


# ===== Definir el request =====
class PredictRequest(BaseModel):
    features: list  # ejemplo: [30, 150, 2]


# ===== Endpoint =====
@app.post("/predict")
def predict(data: PredictRequest):
    X = np.array([data.features])
    pred = model.predict(X)
    return {"prediction": float(pred[0])}

# ===== Endpoint de salud =====
@app.get("/")
def home():
    return {"status": "ok", "message": "API funcionando"}
