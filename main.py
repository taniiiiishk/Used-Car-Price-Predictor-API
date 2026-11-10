from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
CURRENT_YEAR = bundle["current_year"]

app = FastAPI(title="Used Car Price Predictor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CarInput(BaseModel):
    brand: str
    model: str
    year: int = Field(..., ge=1990, le=CURRENT_YEAR)
    km_driven: int = Field(..., ge=0, le=1_000_000)
    fuel: str
    seller_type: str
    transmission: str
    owner: str


@app.get("/")
def home():
    index = BASE_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "API is running. Open /docs to test it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    """Values for the frontend dropdowns."""
    return {k: bundle[k] for k in
            ("brands", "models_by_brand", "fuels", "seller_types", "transmissions", "owners")}


@app.post("/predict")
def predict(car: CarInput):
    checks = {
        "brand": (car.brand, bundle["brands"]),
        "fuel": (car.fuel, bundle["fuels"]),
        "seller_type": (car.seller_type, bundle["seller_types"]),
        "transmission": (car.transmission, bundle["transmissions"]),
        "owner": (car.owner, bundle["owners"]),
    }
    for field, (value, allowed) in checks.items():
        if value not in allowed:
            raise HTTPException(422, f"Invalid {field}: '{value}'. Allowed: {allowed}")
    if car.model not in bundle["models_by_brand"].get(car.brand, []):
        raise HTTPException(422, f"Model '{car.model}' not found for brand '{car.brand}'. See /options.")

    row = pd.DataFrame([{
        "km_driven": car.km_driven,
        "fuel": car.fuel,
        "seller_type": car.seller_type,
        "transmission": car.transmission,
        "owner": car.owner,
        "car_age": CURRENT_YEAR - car.year,
        "brand": car.brand,
        "model": car.model,
    }])
    price = float(model.predict(row)[0])
    return {
        "predicted_price_inr": round(price),
        "range_inr": [round(price * 0.9), round(price * 1.1)],
    }