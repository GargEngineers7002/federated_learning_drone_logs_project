# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.staticfiles import StaticFiles
# from typing import Annotated
# import pandas as pd
# import os

# from ml_models import preprocess_data, run_predictions

# # Debug print
# print("--- STARTING FILE PATH DEBUG ---")
# path_to_html = "template/index.html"
# current_directory = os.getcwd()
# print(f"Current Working Directory is: {current_directory}")
# print(f"Checking: {os.path.join(current_directory, path_to_html)}")
# print(f"Exists? --> {os.path.exists(path_to_html)}")
# print("--- FINISHED FILE PATH DEBUG ---")

# app = FastAPI(title="UAV Trajectory Prediction API")

# @app.post("/api/predict")
# async def predict_trajectory(
#     uav_model: Annotated[str, Form()],
#     flight_log: Annotated[UploadFile, File()]
# ):
#     if not flight_log.filename.endswith('.csv'):
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

#     try:
#         df = pd.read_csv(flight_log.file)

#         preprocessed = preprocess_data(df.copy(), uav_model)

#         results = run_predictions(preprocessed, df)

#         return {"uav_model": uav_model, "results": results}

#     except ValueError as ve:
#         raise HTTPException(status_code=404, detail=str(ve))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# # Serve frontend

# # app.mount("/", StaticFiles(directory="app/template", html=True), name="static")


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Annotated
import pandas as pd
import numpy as np

from ml_models import preprocess_data, run_predictions

app = FastAPI(title="UAV Trajectory Prediction API")


# -----------------------
# TEST ROUTE
# -----------------------
@app.get("/test")
async def test():
    return {"status": "API is running"}


# -----------------------
# POST /api/predict
# -----------------------
@app.post("/api/predict_trajectory")
async def predict_trajectory(
    uav_model: Annotated[str, Form()], flight_log: Annotated[UploadFile, File()]
):
    if not flight_log.filename or not flight_log.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        # 1. Read CSV
        df = pd.read_csv(flight_log.file, low_memory=False)

        # 2. Run Preprocessing
        preprocessed = preprocess_data(df.copy(), uav_model)

        # Clean the Raw Data (df) too
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        df = df.fillna(0)

        # =========================================================

        # 3. Run Predictions (pass uav_model so correct models/scalers are used)
        results = run_predictions(preprocessed, df, uav_model)

        return {"uav_model": uav_model, "results": results}

    except ValueError as ve:
        print(f"PROCESSING ERROR: {ve}")
        raise HTTPException(status_code=400, detail=f"Processing Error: {str(ve)}")
    except Exception as e:
        print(f"GENERAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# -----------------------
# POST /api/upload_results
# -----------------------
@app.post("/api/upload_results")
async def upload_results(data: dict):
    """
    Receives prediction results from federated learning nodes.
    """
    uav_model = data.get("uav_model")
    results = data.get("results")
    
    print(f"Received results from node for {uav_model}")
    return {"status": "success", "message": f"Results for {uav_model} uploaded."}


# -----------------------
# STATIC FILES (FRONTEND)
# -----------------------
app.mount("/", StaticFiles(directory="template", html=True), name="static")
