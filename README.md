# Drone Trajectory Prediction - Central Hub (Backend)

A centralized federated learning backend for aggregating drone trajectory prediction models.

## Architecture Overview

This is the **Central Hub (Backend)** of the federated learning system:
*   **Central Hub (Backend):**
    *   Hosts the FastAPI backend server.
    *   Provides global model weights to clients (`GET /api/get_global`).
    *   Receives updated weights from clients and performs federated averaging (`POST /api/federated_averaging`).

## Project Structure

```text
website_work/
├── app/
│   ├── main.py                 # Central Hub API Server
│   ├── fl.py                   # Global Model aggregation and retrieval logic
│   └── ml_models.py            # Model definition / tools
└── models/                     # Global .keras models and scalers
```

## Setup & Installation

1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Central Hub:**
    The hub runs on port 8000.
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python website_work/app/main.py
    ```
