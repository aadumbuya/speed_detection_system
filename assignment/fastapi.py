from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from src.preprocessing import *
from src.prediction import prediction
from src.retraining import retrained, process

app = FastAPI()

# Define file paths and other configurations
MODEL_NAME = "models/dcl_model.pkl"
SCALER_TYPE = "ss"
DATA_TYPE = "csv"
TRAIN_COLUMN_LIMIT = 10  # Customize based on your dataset
TEST_RATIO = 0.3
ANALYSIS_TYPE = "default"
CLASSES = [1, 0]  # Modify based on your class structure

# Helper function to save uploaded file
def save_upload_file(uploaded_file: UploadFile, destination: str):
    try:
        with open(destination, "wb") as f:
            f.write(uploaded_file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

# Endpoint for making predictions
@app.post("/predict/")
async def predict_endpoint(file: UploadFile, model_name: str = Form(...)):
    try:
        # Save uploaded file
        file_path = f"uploaded_{file.filename}"
        save_upload_file(file, file_path)

        # Load data
        data = pd.read_csv(file_path)

        # Make predictions
        predictions = prediction(data, model_name, SCALER_TYPE)
        os.remove(file_path)  # Clean up the uploaded file
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Endpoint for retraining the model
class RetrainRequest(BaseModel):
    train_data: list  # JSON array or equivalent
    train_labels: list  # JSON array or equivalent
    retrain_type: str  # "only_new" or "new_and_old"

@app.post("/retrain/")
async def retrain_endpoint(request: RetrainRequest):
    try:
        # Convert input data to DataFrame or arrays
        train_df = pd.DataFrame(request.train_data)
        labels = pd.Series(request.train_labels)

        # Process and split data
        train_x, test_x, label_x, test_y = process(
            loader_type="one", 
            train=train_df, 
            label="target_column",  # Replace with your label column
            train_column_limit=TRAIN_COLUMN_LIMIT, 
            scaler_type=SCALER_TYPE,
            data_type=DATA_TYPE,
            test_ratio=TEST_RATIO
        )

        # Retrain the model
        metrics = retrained(
            train=train_x, 
            label=label_x, 
            test=test_x, 
            test_label=test_y, 
            classes=CLASSES, 
            typ_=request.retrain_type, 
            model_name=MODEL_NAME, 
            scaler_type=SCALER_TYPE, 
            analysis_type=ANALYSIS_TYPE
        )
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Endpoint for health check
@app.get("/")
async def root():
    return {"message": "API is running!"}
