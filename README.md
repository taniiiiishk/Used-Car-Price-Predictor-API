# 🚗 Used Car Price Predictor

A machine-learning web app that estimates the resale price of a used car in India, built on CarDekho listings. A FastAPI backend serves the trained model, and a simple web page lets you try it.

## Features
- Predicts price in ₹ from brand, model, year, km driven, fuel, transmission, seller type and owner
- Shows a likely price range (±10%)
- REST API with interactive docs (Swagger) at `/docs`
- Web UI with dependent brand → model dropdowns

## Dataset
CarDekho used car listings (~4,300 rows). After removing duplicates and extreme outliers, about 3,500 rows were used.

## Approach
1. Cleaned data (duplicates, outliers above the 99th percentile)
2. Feature engineering: car age, brand, and model from the car name
3. Log-transformed the price target
4. Compared Linear Regression, Ridge, Random Forest, Gradient Boosting and XGBoost
5. Tuned the best model with 5-fold cross-validation

## Results
| Metric | Value |
|---|---|
| Best model | Ridge Regression |
| R² (test) | ~0.81 |
| MAE | ~₹84,000 |

Adding the car model as a feature improved accuracy more than switching to a complex algorithm. The dataset has no engine or power data, which limits accuracy.

## Tech stack
Python, pandas, scikit-learn, XGBoost, FastAPI, Uvicorn

## Project structure
```
├── data/            # CSV datasets
├── models/          # trained model (car_price_model.pkl)
├── notebooks/       # training notebook
├── index.html       # web UI
├── main.py          # FastAPI app
└── requirements.txt
```

## Run locally
The model was trained with scikit-learn 1.6.1, so use Python 3.12 and the pinned version.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000/ for the web UI or http://127.0.0.1:8000/docs for the API docs.

## API
`POST /predict`
```json
{
  "brand": "Maruti", "model": "Maruti Swift", "year": 2015, "km_driven": 50000,
  "fuel": "Petrol", "seller_type": "Individual",
  "transmission": "Manual", "owner": "First Owner"
}
```
Response:
```json
{"predicted_price_inr": 409003, "range_inr": [368103, 449904]}
```

## Screenshots
Add screenshots of the web UI here.

## Future scope
- Add engine, power and mileage features
- SHAP explanations
- Deploy online (Render / Railway)

## Author
Tanishk, B.Tech CSE, LNCT Bhopal
