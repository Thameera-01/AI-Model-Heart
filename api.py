"""
FastAPI endpoint for heart disease prediction.
This API provides a prediction endpoint that takes user health data
and returns a heart disease risk prediction using the trained model.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import tensorflow as tf
import numpy as np
import joblib
import os
from typing import Optional

# Global variables for model and scalers
model = None
scaler_data = None
scaler_target = None

def load_model_and_scalers():
    """Load the trained model and scalers"""
    global model, scaler_data, scaler_target
    
    try:
        model_path = 'models/heart_model.h5'
        scaler_data_path = 'models/scaler_data.sav'
        scaler_target_path = 'models/scaler_target.sav'
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(scaler_data_path):
            raise FileNotFoundError(f"Scaler data file not found at {scaler_data_path}")
        if not os.path.exists(scaler_target_path):
            raise FileNotFoundError(f"Scaler target file not found at {scaler_target_path}")
        
        # Load model and scalers
        model = tf.keras.models.load_model(model_path)
        scaler_data = joblib.load(scaler_data_path)
        scaler_target = joblib.load(scaler_target_path)
        
        print("Model and scalers loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model and scalers: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    success = load_model_and_scalers()
    if not success:
        print("WARNING: Failed to load model. API will not work properly.")
        print("Please run 'python train_model.py' to create the model files.")
    yield
    # Shutdown (cleanup if needed)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk based on health parameters",
    version="1.0.0",
    lifespan=lifespan
)

# Define input schema using Pydantic
class HealthData(BaseModel):
    """Input schema for health data"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": 0,
                "age": 54,
                "tc": 128.0,
                "hdl": 42.0,
                "smoke": 0,
                "bpm": 1,
                "diabetes": 0
            }
        }
    )
    
    gender: int = Field(..., ge=0, le=1, description="Gender: 0 for Male, 1 for Female")
    age: int = Field(..., ge=1, le=120, description="Age in years")
    tc: float = Field(..., gt=0, description="Total Cholesterol (TC)")
    hdl: float = Field(..., gt=0, description="HDL Cholesterol")
    smoke: int = Field(..., ge=0, le=1, description="Smoking status: 0 for Non-Smoker, 1 for Smoker")
    bpm: int = Field(..., ge=0, le=1, description="High blood pressure: 0 for No, 1 for Yes")
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes: 0 for No, 1 for Yes")

# Define output schema
class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    risk_score: float = Field(..., description="Predicted risk score (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    message: str = Field(..., description="Additional information about the prediction")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    scalers_loaded = scaler_data is not None and scaler_target is not None
    
    return {
        "status": "healthy" if (model_loaded and scalers_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "scalers_loaded": scalers_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(health_data: HealthData):
    """
    Predict heart disease risk based on health parameters.
    
    Parameters:
    - gender: 0 for Male, 1 for Female
    - age: Age in years (1-120)
    - tc: Total Cholesterol
    - hdl: HDL Cholesterol
    - smoke: Smoking status (0 for Non-Smoker, 1 for Smoker)
    - bpm: High blood pressure (0 for No, 1 for Yes)
    - diabetes: Diabetes (0 for No, 1 for Yes)
    
    Returns:
    - risk_score: Predicted risk score (0-1)
    - risk_level: Low, Medium, or High
    - message: Additional information
    """
    # Check if model is loaded
    if model is None or scaler_data is None or scaler_target is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist in the 'models' directory."
        )
    
    try:
        # Prepare input data
        input_data = np.array([
            health_data.gender,
            health_data.age,
            health_data.tc,
            health_data.hdl,
            health_data.smoke,
            health_data.bpm,
            health_data.diabetes
        ]).reshape(1, -1)
        
        # Scale input data
        scaled_data = scaler_data.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data, verbose=0)
        
        # Inverse transform to get actual risk score
        risk_score = float(scaler_target.inverse_transform(prediction)[0][0])
        
        # Ensure risk score is between 0 and 1
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "Low"
            message = "Your heart disease risk is relatively low. Continue maintaining a healthy lifestyle."
        elif risk_score < 0.7:
            risk_level = "Medium"
            message = "Your heart disease risk is moderate. Consider consulting with a healthcare provider."
        else:
            risk_level = "High"
            message = "Your heart disease risk is high. Please consult with a healthcare provider soon."
        
        return PredictionResponse(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("Starting Heart Disease Prediction API...")
    print("API will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
