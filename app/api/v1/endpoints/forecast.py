from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime
import os

router = APIRouter(
    prefix="/api/v1/forecast",
    tags=["forecast"],
    responses={404: {"description": "Not found"}},
)

# Load the pre-trained model (adjust path as needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "demand_forecast_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define the seasons based on your notebook analysis
SEASON_MAP = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}

class ForecastRequest(BaseModel):
    date: str  # Format: YYYY-MM-DD
    store_id: str
    product_id: str
    category: str
    region: str
    inventory_level: int
    units_sold: int
    units_ordered: int
    price: float
    discount: int
    weather_condition: str
    holiday_promotion: int
    competitor_pricing: float

class ForecastResponse(BaseModel):
    forecast: float
    confidence: Optional[float] = None
    message: Optional[str] = None

def determine_season(date_str: str) -> str:
    """Determine season from date based on your notebook's seasonality analysis"""
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_obj.month
    
    for season, months in SEASON_MAP.items():
        if month in months:
            return season
    return "Unknown"

def preprocess_input(data: ForecastRequest) -> pd.DataFrame:
    """Preprocess the input data to match model training format"""
    # Create a DataFrame from the request
    input_dict = data.dict()
    
    # Add derived features based on your notebook
    input_dict["seasonality"] = determine_season(data.date)
    
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Convert date to datetime and extract features (if used in your model)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    
    # Clean column names (replace spaces with underscores if needed)
    df.columns = [col.replace(" ", "_").lower() for col in df.columns]
    
    # Ensure we have all expected columns from your training data
    expected_columns = [
        'date', 'store_id', 'product_id', 'category', 'region', 
        'inventory_level', 'units_sold', 'units_ordered', 'demand_forecast',
        'price', 'discount', 'weather_condition', 'holiday_promotion',
        'competitor_pricing', 'seasonality', 'year', 'month', 'day'
    ]
    
    # Add missing columns with default values if necessary
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Or appropriate default
    
    return df

@router.post("/predict", response_model=ForecastResponse)
async def predict_demand(request: ForecastRequest):
    """
    Predict product demand based on historical sales patterns and current conditions.
    
    Parameters:
    - All fields from your retail_store_inventory.csv that were used in model training
    """
    try:
        # Preprocess the input data
        input_df = preprocess_input(request)
        
        # Make prediction (adjust based on your actual model requirements)
        prediction = model.predict(input_df)
        
        return {
            "forecast": float(prediction[0]),
            "confidence": 0.95,  # Replace with actual confidence if available
            "message": "Successfully generated demand forecast"
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Return information about the trained model"""
    try:
        return {
            "model_type": str(type(model)),
            "features_used": model.feature_name() if hasattr(model, 'feature_name') else "Unknown",
            "training_date": "2022-01-01",  # Replace with actual training date
            "performance_metrics": {
                "rmse": 22.0,  # From your notebook evaluation
                "r2": 0.95    # Example metric
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))