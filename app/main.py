from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
import pandas as pd
import lightgbm as lgb
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Product Demand Forecasting API",
    description="API for predicting product demand based on historical sales data using LightGBM",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model and preprocessing objects
try:
    model = pickle.load(open("model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Failed to load model or encoders: {str(e)}")

# Define input/output models
class PredictionInput(BaseModel):
    date: date
    store_id: str
    product_id: str
    category: str
    region: str
    inventory_level: int
    units_ordered: int
    price: float
    discount: int
    weather_condition: str
    holiday_promotion: int
    competitor_pricing: float
    seasonality: str

class PredictionOutput(BaseModel):
    forecast: float
    confidence: Optional[float] = None  # Could add confidence intervals later

class BatchPredictionInput(BaseModel):
    items: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    forecasts: List[float]

# Helper functions from your notebook
def clean_column_names(df):
    """Clean column names as shown in your notebook"""
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('/', '_')
    return df

def preprocess_data(df):
    """Preprocess the input data similar to your notebook"""
    # Clean column names
    df = clean_column_names(df)
    
    # Convert date to datetime and extract features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Encode categorical variables using the saved label encoders
    categorical_cols = ['store_id', 'product_id', 'category', 'region', 
                       'weather_condition', 'seasonality']
    
    for col in categorical_cols:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
    
    # Drop original date column
    df = df.drop(columns=['date'])
    
    return df

# API endpoints
@app.post("/predict", response_model=PredictionOutput)
async def predict_demand(input_data: PredictionInput):
    """Make a single demand prediction"""
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess the input
        processed_df = preprocess_data(input_df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        
        return {"forecast": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch_demand(input_data: BatchPredictionInput):
    """Make batch predictions"""
    try:
        # Convert input to DataFrame
        input_dicts = [item.dict() for item in input_data.items]
        input_df = pd.DataFrame(input_dicts)
        
        # Preprocess the input
        processed_df = preprocess_data(input_df)
        
        # Make predictions
        predictions = model.predict(processed_df)
        
        return {"forecasts": [float(pred) for pred in predictions]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    """Return information about the trained model"""
    try:
        if isinstance(model, lgb.Booster):
            return {
                "model_type": "LightGBM",
                "num_features": model.num_feature(),
                "num_trees": model.num_trees(),
                "objective": model.params.get('objective', 'unknown')
            }
        else:
            return {"model_type": str(type(model))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Product Demand Forecasting API",
        "endpoints": {
            "/predict": "POST - Make a single prediction",
            "/predict_batch": "POST - Make batch predictions",
            "/model_info": "GET - Get model information"
        }
    }