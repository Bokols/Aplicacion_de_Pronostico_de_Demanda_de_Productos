import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import os
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles all data preprocessing steps for the demand forecasting model"""
    
    def __init__(self):
        self.preprocessor = None
        self.categorical_features = ['store_id', 'product_id', 'category', 
                                   'region', 'weather_condition', 'seasonality']
        self.numerical_features = ['inventory_level', 'units_sold', 'units_ordered', 
                                  'price', 'discount', 'holiday_promotion', 
                                  'competitor_pricing']
        self._initialize_preprocessor()

    def _initialize_preprocessor(self):
        """Initialize the preprocessing pipeline"""
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
    def clean_data(self, df):
        """
        Clean raw data according to the steps in the notebook
        Args:
            df: Raw pandas DataFrame
        Returns:
            Cleaned DataFrame
        """
        try:
            # Clean column names (as shown in notebook)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing values (from notebook analysis)
            df = self._handle_missing_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise

    def _handle_missing_values(self, df):
        """Handle missing values based on notebook analysis"""
        # From notebook: No missing values found, but adding this for robustness
        if df.isnull().sum().sum() > 0:
            logger.info("Handling missing values")
            
            # Numerical columns - fill with median
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            
            # Categorical columns - fill with mode
            cat_cols = df.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df

    def feature_engineering(self, df):
        """
        Add engineered features as shown in the notebook
        Args:
            df: Cleaned DataFrame
        Returns:
            DataFrame with additional features
        """
        try:
            # Extract date components
            if 'date' in df.columns:
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                
                # Add seasonality (from notebook)
                df['seasonality'] = df['date'].apply(self._get_season)
            
            # Add any other features from notebook analysis
            # Example: Price discount ratio
            df['price_discount_ratio'] = df['price'] / (df['discount'] + 1)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def _get_season(self, date):
        """Determine season from date (from notebook)"""
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    def fit_preprocessor(self, df):
        """
        Fit the preprocessor on training data
        Args:
            df: Training DataFrame
        """
        try:
            # Ensure we have all expected columns
            df = self._ensure_columns(df)
            
            # Fit the preprocessor
            self.preprocessor.fit(df)
            
            logger.info("Preprocessor fitted successfully")
            
        except Exception as e:
            logger.error(f"Preprocessor fitting failed: {str(e)}")
            raise

    def transform_data(self, df):
        """
        Transform data using fitted preprocessor
        Args:
            df: DataFrame to transform
        Returns:
            Transformed numpy array
        """
        try:
            # Ensure we have all expected columns
            df = self._ensure_columns(df)
            
            # Transform data
            transformed_data = self.preprocessor.transform(df)
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise

    def _ensure_columns(self, df):
        """Ensure all expected columns are present"""
        # Add missing columns with default values
        for col in self.numerical_features + self.categorical_features:
            if col not in df.columns:
                if col in self.numerical_features:
                    df[col] = 0
                else:
                    df[col] = 'unknown'
        
        return df

    def prepare_inference_data(self, input_dict):
        """
        Prepare API request data for prediction
        Args:
            input_dict: Dictionary from API request
        Returns:
            Processed DataFrame ready for prediction
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_dict])
            
            # Clean data
            df = self.clean_data(df)
            
            # Feature engineering
            df = self.feature_engineering(df)
            
            # Ensure correct column order and presence
            df = self._ensure_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Inference data preparation failed: {str(e)}")
            raise

    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.preprocessor, filepath)
            logger.info(f"Preprocessor saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {str(e)}")
            raise

    def load_preprocessor(self, filepath):
        """Load a saved preprocessor"""
        try:
            self.preprocessor = joblib.load(filepath)
            logger.info(f"Preprocessor loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {str(e)}")
            raise