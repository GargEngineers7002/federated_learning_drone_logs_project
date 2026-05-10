import uuid
import asyncio
import os
import multiprocessing
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Annotated, Dict, List
from contextlib import asynccontextmanager

# Import FL server components
from website_work.app.fl import (
    process_job,
    get_global_model,
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


@app.get("/api/get_global")
async def get_global(uav_model: str):
    print(f"Sending the Global Model {uav_model} to the client Hub.")
    try:
        global_model_weights = await get_global_model(uav_model)
        return global_model_weights
    except Exception as e:
        print(f"[HUB] ERROR in get_global: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


@app.post("/api/federated_averaging")
async def federated_averaging(
    uav_model: Annotated[str, Form()],
    weights: Annotated[str, Form()],
):
    print(f"Doing federated averaging on {uav_model}")
    try:
        await federated_average(uav_model, weights)
        return {"status": "Federated averaging completed."}
    except Exception as e:
        print(f"[HUB] ERROR in federated_averaging: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("\n🔍 Launching Uvicorn server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
