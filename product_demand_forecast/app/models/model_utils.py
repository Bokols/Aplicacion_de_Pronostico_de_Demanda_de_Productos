import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandForecastModel:
    """Wrapper class for the demand forecasting model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.preprocessor = None
        self.features = None
        self.categorical_features = ['store_id', 'product_id', 'category', 
                                   'region', 'weather_condition', 'seasonality']
        self.numerical_features = ['inventory_level', 'units_sold', 'units_ordered', 
                                  'price', 'discount', 'holiday_promotion', 
                                  'competitor_pricing']
        if model_path:
            self.load_model(model_path)

    def build_preprocessor(self):
        """Build the preprocessing pipeline based on the notebook"""
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return self.preprocessor

    def train_model(self, data_path, test_size=0.2, random_state=42):
        """Train the model as shown in the notebook"""
        try:
            # Load and prepare data
            df = pd.read_csv(data_path)
            df = self.clean_data(df)
            
            # Feature engineering
            df = self.feature_engineering(df)
            
            # Define features and target
            X = df.drop('demand_forecast', axis=1)
            y = df['demand_forecast']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            
            # Build and fit preprocessor
            self.build_preprocessor()
            self.preprocessor.fit(X_train)
            
            # Transform data
            X_train_processed = self.preprocessor.transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Train LightGBM model (parameters from your notebook)
            self.model = lgb.LGBMRegressor(
                objective='regression',
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=100,
                random_state=42
            )
            
            self.model.fit(X_train_processed, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_processed, y_train)
            test_score = self.model.score(X_test_processed, y_test)
            
            logger.info(f"Training R2: {train_score:.4f}")
            logger.info(f"Test R2: {test_score:.4f}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def clean_data(self, df):
        """Clean data as shown in the notebook"""
        # Convert column names to lowercase with underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values if any (from notebook analysis)
        if df.isnull().sum().sum() > 0:
            logger.warning("Missing values found, filling with defaults")
            df.fillna({
                'inventory_level': df['inventory_level'].median(),
                'units_sold': 0,
                'units_ordered': 0,
                'price': df['price'].mean(),
                'discount': 0,
                'competitor_pricing': df['competitor_pricing'].mean()
            }, inplace=True)
        
        return df

    def feature_engineering(self, df):
        """Add features from the notebook analysis"""
        # Extract date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        # Add seasonality (from notebook)
        df['seasonality'] = df['date'].apply(self._get_season)
        
        # Add any other engineered features from notebook
        
        return df

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

    def predict(self, input_data):
        """Make prediction on processed input data"""
        try:
            if not self.model:
                raise ValueError("Model not loaded or trained")
            
            # Preprocess input
            processed_data = self.preprocessor.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def save_model(self, output_path):
        """Save model and preprocessor"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'features': self.numerical_features + self.categorical_features,
                'timestamp': datetime.now().isoformat()
            }, output_path)
            logger.info(f"Model saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, model_path):
        """Load saved model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.features = model_data['features']
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

def prepare_input_data(input_dict):
    """Prepare input data for prediction from API request"""
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Ensure all expected columns are present
    expected_cols = [
        'date', 'store_id', 'product_id', 'category', 'region',
        'inventory_level', 'units_sold', 'units_ordered', 'price',
        'discount', 'weather_condition', 'holiday_promotion',
        'competitor_pricing'
    ]
    
    # Add missing columns with default values
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0 if col in ['inventory_level', 'units_sold', 'units_ordered', 
                                  'discount', 'holiday_promotion'] else np.nan
    
    return df