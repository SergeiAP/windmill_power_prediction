# pylint: disable=missing-module-docstring
import os
import sys

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.metrics import mean_absolute_error

# Load the environment varibles from the .env file into the application to get 
# "MLFLOW_S3_ENDPOINT_URL"
load_dotenv()

# Initialize the FastAPI application
app = FastAPI()


# Create a class to store the developed model and predict
class Model:
    """class to upload prefitted model and predict then
    """
    def __init__(self, model_name: str, model_stage: str) -> None:
        """
        Initialize the model
        Args:
            model_name (str): model name in registry
            model_stage (str): model stage
        """
        # Load the model from Registry
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions by uploaded model
        Args:
            data (pd.DataFrame): features for prediction
        """
        predictions: np.ndarray = self.model.predict(data)  # type: ignore
        return predictions
    
# Create model
model = Model("wpp_selector-ridge", "Staging")

# Create the POST endpoint with path '/invocations'
@app.post("/invocations")
async def predict(file: UploadFile = File(...)) -> list[str]:
    """Generate predicitions using file

    Args:
        file (UploadFile, optional): uploaded file with features.
        Defaults to File(...).

    Raises:
        HTTPException: if any problem with file

    Returns:
        list[str]: predictions
    """
    # Handle the file in csv-only format
    if file.filename.endswith(".csv"):
        # Create a temprorary file to load the data into a pd.DataFrame
        with open(file.filename, "wb") as file_:
            file_.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        # Return a JSON object containing the model predicitons
        return list(model.predict(data))
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(
            status_code=400, detail="Invalid file format. CSV-only files accpeted.")

@app.post("/test")
async def test_predictions(file: UploadFile = File(...)) -> dict:
    """Get MAE and target + predictions

    Args:
        file (UploadFile, optional): uploaded file with features.
        Defaults to File(...).

    Raises:
        HTTPException: if any problem with file

    Returns:
        dict: target and predictions
    """
    # Handle the file in csv-only format
    if file.filename.endswith(".csv"):
        # Create a temprorary file to load the data into a pd.DataFrame
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        # Return a JSON object containing the model predicitons
        df_pred = pd.concat([data["date"], data["wp"]], axis="columns")
        data.drop(["date", "wp"], axis="columns", inplace=True)
        df_pred["predictions"] = model.predict(data)
        print(mean_absolute_error(df_pred["wp"], df_pred["predictions"]))
        
        return df_pred.to_dict(orient='list')
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(
            status_code=400, detail="Invalid file format. CSV-only files accpeted.")

# Check if the environment varibales for AWS access are available
# If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    sys.exit(1)
