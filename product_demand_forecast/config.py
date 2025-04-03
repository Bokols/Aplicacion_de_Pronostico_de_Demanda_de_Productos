import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    APP_NAME: str = os.getenv("APP_NAME", "Product Demand Forecasting API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    APP_ENV: str = os.getenv("APP_ENV", "production")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "False").lower() == "true"
    
    # Path Configuration
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_PATH: Path = BASE_DIR / os.getenv("MODEL_PATH", "app/model.pkl")
    ENCODERS_PATH: Path = BASE_DIR / os.getenv("ENCODERS_PATH", "app/label_encoders.pkl")
    DATA_PATH: Path = BASE_DIR / os.getenv("DATA_PATH", "app/retail_store_inventory.csv")
    
    # Model Configuration
    MODEL_TYPE: str = "lightgbm"
    FEATURES: list = [
        'store_id', 'product_id', 'category', 'region', 
        'inventory_level', 'units_ordered', 'price', 
        'discount', 'weather_condition', 'holiday_promotion',
        'competitor_pricing', 'seasonality', 'year', 
        'month', 'day', 'day_of_week'
    ]
    CATEGORICAL_FEATURES: list = [
        'store_id', 'product_id', 'category', 
        'region', 'weather_condition', 'seasonality'
    ]
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            "model_type": cls.MODEL_TYPE,
            "features": cls.FEATURES,
            "categorical_features": cls.CATEGORICAL_FEATURES,
            "model_path": str(cls.MODEL_PATH)
        }
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration"""
        if not cls.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {cls.MODEL_PATH}")
        if not cls.ENCODERS_PATH.exists():
            raise FileNotFoundError(f"Encoders file not found at {cls.ENCODERS_PATH}")
        
        if cls.APP_ENV not in ["development", "production", "testing"]:
            raise ValueError(f"Invalid APP_ENV: {cls.APP_ENV}")

class DevelopmentConfig(Config):
    """Development specific configuration"""
    API_RELOAD = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production specific configuration"""
    API_RELOAD = False
    LOG_LEVEL = "WARNING"

class TestingConfig(Config):
    """Testing specific configuration"""
    API_RELOAD = False
    LOG_LEVEL = "DEBUG"
    MODEL_PATH = Path("tests/test_model.pkl")
    ENCODERS_PATH = Path("tests/test_encoders.pkl")

def get_config() -> Config:
    """Get appropriate config class based on environment"""
    env = os.getenv("APP_ENV", "production").lower()
    if env == "development":
        return DevelopmentConfig()
    elif env == "testing":
        return TestingConfig()
    return ProductionConfig()

# Initialize config
config = get_config()
config.validate_config()