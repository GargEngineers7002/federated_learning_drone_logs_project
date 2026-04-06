import uuid
import asyncio
import os
import multiprocessing
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Annotated, Dict, List
from contextlib import asynccontextmanager

# Import FL server components
from website_work.app.fl import (
    process_job,
    get_global_model,
    get_processed_data,
    federated_average,
)

# Suppress TensorFlow GPU warnings if CPU-only is expected
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


app = FastAPI(title="UAV Trajectory Prediction Central Hub")


# -----------------------
# TEST ROUTE
# -----------------------
@app.get("/test")
async def test():
    print("DEBUG: /test endpoint pinged.")
    return {"status": "Central Hub is running"}


# -----------------------
# POST /api/predict_trajectory
# -----------------------
@app.post("/api/predict_trajectory")
async def predict_trajectory(
    uav_model: Annotated[str, Form()], flight_log: Annotated[UploadFile, File()]
):
    print(f"\n[USER] New prediction request received for model: {uav_model}")
    if not flight_log.filename or not flight_log.filename.lower().endswith(".csv"):
        print("[USER] Error: Invalid file type uploaded.")
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        # 1. Read CSV as string to send to node
        contents = await flight_log.read()
        csv_str = contents.decode("utf-8")

        job_id = str(uuid.uuid4())  # Generate a unique job ID

        results = await process_job(job_id, uav_model, csv_str)  # Run prediction

        print(f"[HUB] Returning results for Job {job_id} to user.")
        return {
            "uav_model": uav_model,
            "results": results,
            "job_id": job_id,
        }

    except Exception as e:
        print(f"[HUB] ERROR in predict_trajectory: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.post("/api/get_global")
async def get_global(uav_model: Annotated[str, Form()]):
    try:
        global_model_weights = await get_global_model(uav_model)
        return global_model_weights
    except Exception as e:
        print(f"[HUB] ERROR in get_global: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.post("/api/get_processed")
async def get_processed(
    uav_model: Annotated[str, Form()], flight_log: Annotated[UploadFile, File()]
):
    try:
        processed_data = await get_processed_data(uav_model, flight_log)
        return processed_data
    except Exception as e:
        print(f"[HUB] ERROR in get_processed: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.post("/api/federated_averaging")
async def federated_averaging(
    uav_model: Annotated[str, Form()],
    weights: Annotated[str, Form()],
):
    try:
        await federated_average(uav_model, weights)
        return {"status": "Federated averaging completed."}
    except Exception as e:
        print(f"[HUB] ERROR in federated_averaging: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# -----------------------
# STATIC FILES (FRONTEND)
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount(
    "/",
    StaticFiles(directory=os.path.join(BASE_DIR, "template"), html=True),
    name="static",
)

if __name__ == "__main__":
    import uvicorn

    print("\n🔍 Launching Uvicorn server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
