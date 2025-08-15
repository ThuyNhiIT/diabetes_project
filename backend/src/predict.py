import pandas as pd
import joblib
from pathlib import Path

# ----------------------------
# Đường dẫn
# ----------------------------
MODEL_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = MODEL_DIR / "xgb_diabetes_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"

# ----------------------------
# Load model, scaler, feature columns
# ----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
print("✅ Model, scaler, feature columns loaded")

# ----------------------------
# Dữ liệu mới (có thể thay bằng pd.read_csv để predict nhiều bệnh nhân)
# ----------------------------
data_new = pd.DataFrame([{
    'year': 2016,
    'gender': 'female',
    'age': 64,
    'location': 'Alabama',
    'race:AfricanAmerican': 0,
    'race:Asian': 0,
    'race:Caucasian': 0,
    'race:Hispanic': 0,
    'race:Other': 1,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'ever',
    'bmi': 49.27,
    'hbA1c_level': 8.2,
    'blood_glucose_level': 140
}])


# ----------------------------
# Xử lý categorical giống train
# ----------------------------
categorical_cols = ['gender', 'location', 'smoking_history']
for col in categorical_cols:
    data_new[col] = data_new[col].astype(str).str.strip().str.lower()

# One-hot encode
data_new = pd.get_dummies(data_new, columns=categorical_cols, dummy_na=True)

# ----------------------------
# Bổ sung cột thiếu và sắp xếp
# ----------------------------
for col in feature_columns:
    if col not in data_new.columns:
        data_new[col] = 0
data_new = data_new[feature_columns]

# ----------------------------
# Chuẩn hóa numeric
# ----------------------------
X_new_scaled = scaler.transform(data_new)

# ----------------------------
# Dự đoán
# ----------------------------
pred_label = model.predict(X_new_scaled)
pred_prob = model.predict_proba(X_new_scaled)[:, 1]

# ----------------------------
# Hiển thị kết quả
# ----------------------------
data_new['Prediction'] = pred_label
data_new['Probability(%)'] = pred_prob * 100

print("\n📊 Prediction Result:")
print(data_new[['Prediction', 'Probability(%)']])

# ----------------------------
# Nếu muốn, lưu kết quả ra CSV
# ----------------------------
OUTPUT_PATH = Path(__file__).parents[1] / "output_predictions.csv"
data_new.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Predictions saved to {OUTPUT_PATH}")
