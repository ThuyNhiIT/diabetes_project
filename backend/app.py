from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# ===================
# Load model + scaler + feature columns
# ===================
MODEL_DIR = Path(__file__).parent / "models"
model = joblib.load(MODEL_DIR / "xgb_diabetes_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

# ===================
# FastAPI init
# ===================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================
# Request schema
# ===================
class PatientData(BaseModel):
    year: int
    gender: str
    age: int
    location: str
    race_AfricanAmerican: int
    race_Asian: int
    race_Caucasian: int
    race_Hispanic: int
    race_Other: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hbA1c_level: float
    blood_glucose_level: float

# ===================
# Predict endpoint
# ===================
@app.post("/predict")
def predict(data: PatientData):
    # Convert sang DataFrame
    df = pd.DataFrame([data.dict()])

    # Chuẩn hóa chữ thường
    categorical_cols = ['gender', 'location', 'smoking_history']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)

    # Thêm cột thiếu & sắp xếp đúng thứ tự
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    pred_label = int(model.predict(X_scaled)[0])
    pred_prob = float(model.predict_proba(X_scaled)[:, 1][0])

    return {
        "prediction": pred_label,
        "probability": pred_prob
    }
