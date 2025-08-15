import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ----------------------------
# Đường dẫn
# ----------------------------
DATA_PATH = Path(__file__).parents[1] / "data" / "diabetes.csv"
MODEL_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = MODEL_DIR / "xgb_diabetes_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"

# ----------------------------
# Log bắt đầu
# ----------------------------
print("📌 Starting training script...")

# ----------------------------
# Load dữ liệu
# ----------------------------
print(f"📌 Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Data shape: {df.shape}")

# ----------------------------
# Xử lý missing values
# ----------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)  # numeric
df.fillna('no info', inplace=True)                  # categorical
print("✅ Missing values handled")

# ----------------------------
# Xử lý cột categorical
# ----------------------------
categorical_cols = ['gender', 'location', 'smoking_history']  # thêm cột khác nếu cần
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
df = pd.get_dummies(df, columns=categorical_cols, dummy_na=True)
print(f"✅ Categorical columns encoded: {categorical_cols}")

# ----------------------------
# Tách features và target
# ----------------------------
X = df.drop('diabetes', axis=1)
y = df['diabetes']
print(f"✅ Features shape: {X.shape}, Target shape: {y.shape}")

# ----------------------------
# Lưu danh sách cột features
# ----------------------------
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, FEATURES_PATH)
print(f"✅ Feature columns saved to {FEATURES_PATH}")

# ----------------------------
# Chuẩn hóa dữ liệu
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Numeric features scaled")

# ----------------------------
# Chia train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ----------------------------
# Train XGBoost
# ----------------------------
print("📌 Training XGBoost model...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)
print("✅ Model training completed")

# ----------------------------
# Dự đoán và đánh giá
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ----------------------------
# Lưu model và scaler
# ----------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Scaler saved to {SCALER_PATH}")

print("🎉 Training script finished successfully!")
