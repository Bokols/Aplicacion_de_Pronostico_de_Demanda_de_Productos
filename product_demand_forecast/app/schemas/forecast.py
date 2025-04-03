from pydantic import BaseModel
from typing import Optional
from datetime import date

class ForecastInput(BaseModel):
    date: str
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
    
    class Config:
        schema_extra = {
            "example": {
                "date": "2022-01-01",
                "store_id": "S001",
                "product_id": "P0001",
                "category": "Groceries",
                "region": "North",
                "inventory_level": 231,
                "units_ordered": 55,
                "price": 33.50,
                "discount": 20,
                "weather_condition": "Rainy",
                "holiday_promotion": 0,
                "competitor_pricing": 29.69,
                "seasonality": "Autumn"
            }
        }

class BatchForecastInput(BaseModel):
    items: list[ForecastInput]

class ForecastOutput(BaseModel):
    demand_forecast: float
    confidence_interval: Optional[tuple[float, float]]
    model_version: str