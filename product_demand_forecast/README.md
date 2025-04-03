# Product Demand Forecasting API

## Overview
This API provides demand forecasting predictions using a LightGBM model trained on retail inventory data.

## API Documentation
Interactive documentation is available at `/docs` when the API is running.

## Endpoints
- `/api/v1/predict` - Single item prediction
- `/api/v1/batch_predict` - Batch predictions

## Deployment
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload