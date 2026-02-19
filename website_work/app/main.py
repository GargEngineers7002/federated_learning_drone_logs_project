import uuid
import asyncio
import io
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Annotated, Dict, List
import pandas as pd
import numpy as np
import os
import threading

# Import FL server components
from website_work.app.federated_learning.fl_server import run_fl_server

app = FastAPI(title="UAV Trajectory Prediction Central Hub")

# -----------------------
# IN-MEMORY JOB STORE
# -----------------------
# jobs = { job_id: { "status": "pending/completed", "uav_model": "...", "data": csv_str, "results": None } }
jobs: Dict[str, dict] = {}
job_queue: List[str] = []

# Start FL Server in background
@app.on_event("startup")
async def startup_event():
    # Run Flower server in a separate thread
    fl_thread = threading.Thread(target=run_fl_server, daemon=True)
    fl_thread.start()
    print("FL Server started in background thread.")

# -----------------------
# TEST ROUTE
# -----------------------
@app.get("/test")
async def test():
    return {"status": "Central Hub is running"}

# -----------------------
# POST /api/predict_trajectory
# -----------------------
@app.post("/api/predict_trajectory")
async def predict_trajectory(
    uav_model: Annotated[str, Form()], flight_log: Annotated[UploadFile, File()]
):
    if not flight_log.filename or not flight_log.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        # 1. Read CSV as string to send to node
        contents = await flight_log.read()
        csv_str = contents.decode('utf-8')
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "pending",
            "uav_model": uav_model,
            "data": csv_str,
            "results": None
        }
        job_queue.append(job_id)
        
        print(f"Queued job {job_id} for {uav_model}")

        # 2. Wait for a node to process it
        timeout = 120 # seconds
        start_time = asyncio.get_event_loop().time()
        while jobs[job_id]["status"] == "pending":
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise HTTPException(status_code=504, detail="Processing timeout. No nodes available.")
            await asyncio.sleep(1)
        
        return {"uav_model": uav_model, "results": jobs[job_id]["results"], "job_id": job_id}

    except Exception as e:
        print(f"GENERAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

# -----------------------
# NODE ENDPOINTS (Internal)
# -----------------------

@app.get("/api/node/get_job")
async def get_job():
    """Nodes call this to get a job."""
    if not job_queue:
        return {"job": None}
    
    job_id = job_queue.pop(0)
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "uav_model": job["uav_model"],
        "data": job["data"]
    }

@app.post("/api/node/submit_results")
async def submit_results(payload: dict):
    """Nodes call this to return prediction results."""
    job_id = payload.get("job_id")
    results = payload.get("results")
    
    if job_id in jobs:
        jobs[job_id]["results"] = results
        jobs[job_id]["status"] = "completed"
        print(f"Job {job_id} completed by node.")
        return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="Job not found")

# -----------------------
# STATIC FILES (FRONTEND)
# -----------------------
app.mount("/", StaticFiles(directory="template", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
