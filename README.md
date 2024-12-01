# Machine Learning Prediction and Retraining API

This repository contains a **FastAPI-based backend system** designed for:
1. **Making predictions**: Users can upload their datasets to generate predictions using a pre-trained machine learning model.
2. **Retraining the model**: Users can upload their own training data to improve or customize the model's performance.

The system is designed to integrate seamlessly with a frontend application, where users interact with endpoints to make predictions or retrain the model.

## Features

### 1. **Prediction Endpoint (`/predict/`)**
- **Purpose**: Allows users to upload datasets for prediction.
- **Process**:
  - Accepts a `.csv` file uploaded by the user.
  - Reads and preprocesses the data.
  - Makes predictions using the pre-trained model.
  - Returns the predictions as a response.

### 2. **Retraining Endpoint (`/retrain/`)**
- **Purpose**: Enables users to retrain the machine learning model with their own data.
- **Process**:
  - Accepts:
    - A training dataset (list format).
    - Corresponding labels for the dataset.
    - The retraining type (`"only_new"` to train with only the uploaded data, or `"new_and_old"` to combine with existing data).
  - Processes and splits the dataset into training and testing subsets.
  - Retrains the model and evaluates it on the test data.
  - Returns the performance metrics of the retrained model.

### 3. **Health Check Endpoint (`/`)**
- **Purpose**: A basic endpoint to check if the API is running.
- **Response**: `{"message": "API is running!"}`

## How It Works

1. **Backend**:
   - Built with FastAPI, a modern, fast web framework for Python.
   - Uses machine learning functionalities from integrated modules (`model.py`, `prediction.py`, `preprocessing.py`, and `retraining.py`).

2. **Model Functionalities**:
   - Preprocessing:
     - Data reading and writing (`CSV` format).
     - Label encoding and scaling.
     - Splitting data into training and testing subsets.
   - Prediction:
     - Utilizes a pre-trained machine learning model to make predictions.
   - Retraining:
     - Retrains the model with new data (optionally includes old data).

3. **Frontend**:
   - Connects with the API endpoints, providing an intuitive interface for uploading datasets and visualizing results.


## API Endpoints

### 1. **Prediction**
- **URL**: `/predict/`
- **Method**: `POST`
- **Parameters**:
  - `file`: The CSV file containing input data.
  - `model_name`: Name of the pre-trained model to use for predictions.
- **Response**:
  - JSON containing predicted outputs.

### 2. **Retraining**
- **URL**: `/retrain/`
- **Method**: `POST`
- **Request Body** (JSON):
  ```json
  {
    "train_data": [[...], [...], ...],  // List of input training data
    "train_labels": [...],             // Corresponding labels
    "retrain_type": "only_new"         // Retrain type: "only_new" or "new_and_old"
  }
  ```
- **Response**:
  - JSON containing evaluation metrics (accuracy, precision, recall, etc.).

### 3. **Health Check**
- **URL**: `/`
- **Method**: `GET`
- **Response**:
  - `{"message": "API is running!"}`


## Setup and Installation

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run the Application**
```bash
uvicorn main:app --reload
```

The application will be accessible at `http://127.0.0.1:8000`.

### 4. **Test the Endpoints**
- Use a tool like Postman or connect the API to your frontend.
- For predictions, upload a CSV file.
- For retraining, upload JSON-formatted data.

## Directory Structure

```
.
├── main.py                   # FastAPI application
├── model.py                  # Model-related functions
├── prediction.py             # Prediction-related functions
├── preprocessing.py          # Data preprocessing utilities
├── retraining.py             # Retraining functions
├── requirements.txt          # Dependencies
└── README.md                 # Documentation (this file)
```

## Technologies Used

- **Python**
- **FastAPI**
- **scikit-learn** (for machine learning model handling)
- **pandas** (for data manipulation)
- **Joblib** (for model persistence)

## Future Enhancements

1. **Enhanced Frontend Integration**:
   - Real-time visualizations for predictions and metrics.

2. **Support for Multiple Models**:
   - Extend API to handle multiple models dynamically.

3. **Model Versioning**:
   - Track changes in retrained models for better management.

---

Model Description:


The classification of accident severity using a traditional machine learning model such as Decision Tree,RandomForest,e.t.c.

The model is designed to take in the dataset,covert all objects to int,split it into training and testing data and then scale each of them using any good scaler such as Standard Scaler,RobustScaler,e.t.c.,then it feeds the scaled training data into the model and calculate the model performance using the test data.


We run the prediction by feeding the model with the features needed for prediction and it gives the severity of the accident.

Retraining takes place by sending in the needed training data,convert to int,split,scale and retrain the trained model,run an evaluation using the test data and giving the report of the model performance.

So we have the train,predict and retrain endpoint for all those tasks and each of them are well written,tested and working perfectly well.