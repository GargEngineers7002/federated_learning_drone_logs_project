import uuid
import asyncio
import io
import json
import os
import multiprocessing
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from typing import Annotated, Dict, List
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

# Suppress TensorFlow GPU warnings if CPU-only is expected
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import FL server components
from website_work.app.federated_learning.fl_server import run_fl_server


# -----------------------
# IN-MEMORY JOB STORE
# -----------------------
jobs: Dict[str, dict] = {}
job_queue: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("\n" + "="*50)
    print("🚀 STARTING CENTRAL HUB (CPU Mode)")
    print("="*50)
    
    # Use multiprocessing instead of threading because Flower's start_server
    # registers signal handlers which only work in the main thread.
    fl_process = multiprocessing.Process(target=run_fl_server, daemon=True)
    fl_process.start()
    print("✅ FL Server (Flower) started in background process.")

    yield  # The FastAPI app runs during this yield

    # --- Shutdown Logic ---
    if fl_process.is_alive():
        fl_process.terminate()
    print("\n" + "="*50)
    print("🛑 SHUTTING DOWN CENTRAL HUB")
    print("="*50)


app = FastAPI(title="UAV Trajectory Prediction Central Hub", lifespan=lifespan)


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

        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "pending",
            "uav_model": uav_model,
            "data": csv_str,
            "results": None,
        }
        job_queue.append(job_id)

        print(f"[HUB] Job created with ID: {job_id}")
        print(f"[HUB] Current Queue Size: {len(job_queue)}")

        # 2. Wait for a node to process it
        timeout = 120  # seconds
        start_time = asyncio.get_event_loop().time()
        while jobs[job_id]["status"] == "pending":
            if asyncio.get_event_loop().time() - start_time > timeout:
                print(f"[HUB] TIMEOUT: Job {job_id} expired waiting for node.")
                raise HTTPException(
                    status_code=504, detail="Processing timeout. No nodes available."
                )
            await asyncio.sleep(1)

        print(f"[HUB] Returning results for Job {job_id} to user.")
        return {
            "uav_model": uav_model,
            "results": jobs[job_id]["results"],
            "job_id": job_id,
        }

    except Exception as e:
        print(f"[HUB] ERROR in predict_trajectory: {e}")
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
    print(f"\n[NODE] Node CONNECTED. Handing off Job {job_id} ({job['uav_model']})")
    return {"job_id": job_id, "uav_model": job["uav_model"], "data": job["data"]}


@app.post("/api/node/submit_results")
async def submit_results(payload: dict):
    """Nodes call this to return prediction results."""
    job_id = payload.get("job_id")
    results = payload.get("results")

    if job_id in jobs:
        jobs[job_id]["results"] = results
        jobs[job_id]["status"] = "completed"
        print(f"[NODE] SUCCESS: Node returned results for Job {job_id}")
        return {"status": "success"}

    print(f"[NODE] ERROR: Node tried to submit results for UNKNOWN Job {job_id}")
    raise HTTPException(status_code=404, detail="Job not found")


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
