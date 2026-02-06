# Heart Disease Prediction API

This document provides instructions for using the FastAPI endpoint for heart disease prediction.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train_model.py
   ```
   This will create the necessary model files in the `models/` directory:
   - `heart_model.h5` - The trained Keras model
   - `scaler_data.sav` - Data scaler
   - `scaler_target.sav` - Target scaler

3. **Start the API server:**
   ```bash
   python api.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL:** `GET /`
- **Description:** Get API information

### 2. Health Check
- **URL:** `GET /health`
- **Description:** Check if the API and model are loaded correctly

### 3. Prediction Endpoint
- **URL:** `POST /predict`
- **Description:** Get heart disease risk prediction
- **Request Body:**
  ```json
  {
    "gender": 0,
    "age": 54,
    "tc": 128.0,
    "hdl": 42.0,
    "smoke": 0,
    "bpm": 1,
    "diabetes": 0
  }
  ```
- **Response:**
  ```json
  {
    "risk_score": 0.4523,
    "risk_level": "Medium",
    "message": "Your heart disease risk is moderate. Consider consulting with a healthcare provider."
  }
  ```

### Input Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| gender | int | 0-1 | Gender (0: Male, 1: Female) - Binary field based on training data |
| age | int | 1-120 | Age in years |
| tc | float | >0 | Total Cholesterol |
| hdl | float | >0 | HDL Cholesterol |
| smoke | int | 0-1 | Smoking status (0: Non-Smoker, 1: Smoker) |
| bpm | int | 0-1 | High blood pressure (0: No, 1: Yes) |
| diabetes | int | 0-1 | Diabetes (0: No, 1: Yes) |

**Note:** The gender field is binary due to the structure of the training data. In production, this should be updated to support more inclusive gender representations based on the available medical data.

### Risk Levels

- **Low (< 0.3):** Low risk, maintain healthy lifestyle
- **Medium (0.3 - 0.7):** Moderate risk, consider consulting healthcare provider
- **High (> 0.7):** High risk, please consult healthcare provider soon

## Interactive Documentation

FastAPI provides automatic interactive API documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Example Usage

### Using cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 0,
    "age": 54,
    "tc": 128.0,
    "hdl": 42.0,
    "smoke": 0,
    "bpm": 1,
    "diabetes": 0
  }'
```

### Using Python:
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "gender": 0,
    "age": 54,
    "tc": 128.0,
    "hdl": 42.0,
    "smoke": 0,
    "bpm": 1,
    "diabetes": 0
}

response = requests.post(url, json=data)
result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Message: {result['message']}")
```

### Using JavaScript (fetch):
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    gender: 0,
    age: 54,
    tc: 128.0,
    hdl: 42.0,
    smoke: 0,
    bpm: 1,
    diabetes: 0
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Notes

- The model files are not included in the repository due to their size
- Run `train_model.py` to generate the model files locally
- The training script uses synthetic data for demonstration
- For production use, replace the synthetic data with actual medical data
