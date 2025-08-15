from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

# Load model, scaler, feature columns
MODEL_DIR = Path("models")
model = joblib.load(MODEL_DIR / "xgb_diabetes_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

app = FastAPI()

# --- Cấu hình CORS ---
origins = [
    "http://localhost:3000",  # frontend React
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Cho phép các domain này gọi API
    allow_credentials=True,
    allow_methods=["*"],        # Cho phép GET, POST,...
    allow_headers=["*"],        # Cho phép mọi header
)

# Cấu trúc dữ liệu bệnh nhân
class Patient(BaseModel):
    year: int
    gender: str
    age: int
    location: str
    race_AfricanAmerican: int = 0
    race_Asian: int = 0
    race_Caucasian: int = 0
    race_Hispanic: int = 0
    race_Other: int = 0
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hbA1c_level: float
    blood_glucose_level: float

@app.post("/predict")
def predict_diabetes(patient: Patient):
    data_new = pd.DataFrame([patient.dict()])

    # Xử lý categorical giống training
    for col in ['gender', 'location', 'smoking_history']:
        data_new[col] = data_new[col].astype(str).str.strip().str.lower()
    data_new = pd.get_dummies(data_new, columns=['gender','location','smoking_history'], dummy_na=True)

    # Bổ sung cột thiếu
    for col in feature_columns:
        if col not in data_new.columns:
            data_new[col] = 0
    data_new = data_new[feature_columns]

    # Chuẩn hóa
    X_new_scaled = scaler.transform(data_new)

    # Dự đoán
    pred_label = int(model.predict(X_new_scaled)[0])
    pred_prob = float(model.predict_proba(X_new_scaled)[0][1])

    return {"prediction": pred_label, "probability": pred_prob}
