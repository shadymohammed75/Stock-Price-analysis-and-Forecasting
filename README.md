# Stock Prediction Using Machine Learning

## 📌 Project Overview
This project applies **machine learning models** to predict stock closing prices using historical stock exchange data. The model utilizes **Random Forest Regressor**, trained with past price movements and technical indicators.

### 🚀 **Features of the Model**
- Predicts **next-day stock closing price**.
- Uses **historical stock data** (lags, moving averages, daily returns).
- Implements **Random Forest with hyperparameter tuning**.
- Deployable as an API using **FastAPI** (Feature Work).

---

## 📂 **Dataset & Features**
The dataset includes historical stock prices with the following key features:

### 🔹 **Input Features**
The model takes in **five features** as input:
1. **`Close_Lag1`** – The closing price from 1 day ago.
2. **`Close_Lag2`** – The closing price from 2 days ago.
3. **`Close_Lag3`** – The closing price from 3 days ago.
4. **`30_day_ma1`** – The 30-day moving average of the closing price.
5. **`Daily_Return`** – The daily percentage return.

📌 **Example Input JSON:**
```json
{
    "close_lag1": 150.5,
    "close_lag2": 148.2,
    "close_lag3": 145.8,
    "ma_30": 155.0,
    "daily_return": 0.0023
}
```

### 🔹 **Output Prediction**
- The model predicts the **next day's closing price** based on the input.

📌 **Example Output JSON:**
```json
{
    "Predicted_Close_Price": 152.75
}
```

---

## 📊 **Model Analysis & Results**
### **1️⃣ Initial Model Performance**
- Used **XGBoost and Random Forest** models for prediction.
- Initial **Random Forest RMSE**: `193.71`, **MAE**: `140.82` (High error).

### **2️⃣ Hyperparameter Tuning & Feature Engineering**
- Optimized **Random Forest parameters**:
  ```python
  {
      "max_depth": 20,
      "min_samples_leaf": 2,
      "min_samples_split": 2,
      "n_estimators": 100
  }
  ```
- Added more features: **Moving Averages, Daily Returns**.
- **Tuned RMSE**: `51.64`, **Tuned MAE**: `28.93` (Significant improvement!).

### **3️⃣ Next Steps**
- ✅ Experiment with **LightGBM, LSTM models** for further improvements.
- ✅ Deploy API to **Cloud Platforms (AWS, GCP, Azure)**.
- ✅ Add **real-time stock data updates**.

---

## 🔧 **How to Use the Model**

### **1️⃣ Train & Save the Model**
Run the following script to train and save the model:
```python
import joblib
best_rf_model = RandomForestRegressor(max_depth=20, min_samples_leaf=2, min_samples_split=2, n_estimators=100, random_state=42)
best_rf_model.fit(X_train, y_train)
joblib.dump(best_rf_model, "best_random_forest_model.pkl")
```

### **2️⃣ Load & Make Predictions**
```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_random_forest_model.pkl")

# Create sample input
input_data = np.array([[150.5, 148.2, 145.8, 155.0, 0.0023]])

# Predict next closing price
predicted_price = model.predict(input_data)
print(predicted_price)
```

---

## 🌍 **FastAPI Deployment (Feature Work)**
### **1️⃣ Install Dependencies**
```bash
pip install fastapi uvicorn joblib scikit-learn numpy pandas
```

### **2️⃣ Create FastAPI Server (`app.py`)**
```python
from fastapi import FastAPI
import joblib
import numpy as np

# Load model
model = joblib.load("best_random_forest_model.pkl")
app = FastAPI()

@app.post("/predict/")
def predict(close_lag1: float, close_lag2: float, close_lag3: float, ma_30: float, daily_return: float):
    input_data = np.array([[close_lag1, close_lag2, close_lag3, ma_30, daily_return]])
    predicted_price = model.predict(input_data)[0]
    return {"Predicted_Close_Price": round(predicted_price, 2)}
```

### **3️⃣ Run the API**
```bash
uvicorn app:app --reload
```
API will be available at: **http://127.0.0.1:8000**

### **4️⃣ Test the API (Python Request)**
```python
import requests

response = requests.post("http://127.0.0.1:8000/predict/", json={
    "close_lag1": 150.5,
    "close_lag2": 148.2,
    "close_lag3": 145.8,
    "ma_30": 155.0,
    "daily_return": 0.0023
})
print(response.json())
```

---

## 📌 **Next Steps**
- ✅ **Optimize Model Further** (Try LightGBM, XGBoost)
- ✅ **Deploy API on Cloud** (AWS, GCP, or Azure)
- ✅ **Add Real-Time Data Integration**

📢 **Contributions Welcome!** 🚀

