# Federated Learning Drone Trajectory Prediction

A centralized federated learning system for predicting drone trajectories based on flight logs. This project uses **FastAPI** for the central hub and **Flower (flwr)** for the federated learning orchestration.

## Architecture Overview

The system is designed with a **Hub-and-Spoke** architecture where nodes perform both inference and training:

*   **Central Hub (Server):** 
    *   Hosts the web interface for end users.
    *   Manages a job queue for trajectory prediction.
    *   Runs the Flower Federated Learning server to aggregate model weights.
*   **Worker Node (Client):** 
    *   Polls the Hub for new prediction jobs.
    *   Performs local inference using pre-trained models.
    *   Returns results to the Hub for user visualization.
    *   Participates in Federated Learning rounds to improve the global model without sharing raw data.

## Project Structure

```text
website_work/
├── app/
│   ├── main.py                 # Central Hub API & Web Server
│   ├── ml_models.py            # Model loading and inference logic
│   ├── federated_learning/
│   │   ├── fl_server.py        # Flower FL Server implementation
│   │   ├── fl_client.py        # Worker Node polling & training logic
│   │   └── utils.py            # FL helper functions (model creation, etc.)
│   └── template/               # Frontend (HTML/JS/CSS)
└── models/                     # Pre-trained .keras models and scalers
```

## Setup & Installation

1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Central Hub:**
    The hub starts both the Web API (port 8000) and the FL Server (port 8080).
    ```bash
    python website_work/app/main.py
    ```

3.  **Start a Worker Node:**
    Run this on the machine(s) equipped with the models and training data.
    ```bash
    python website_work/app/federated_learning/fl_client.py --website_url http://localhost:8000 --fl_server localhost:8080
    ```

## Usage Workflow

1.  **User Upload:** An end user uploads a drone flight log (CSV) via the web interface.
2.  **Task Queuing:** The Hub receives the file, creates a job, and adds it to the queue.
3.  **Node Processing:** A Worker Node pulls the job, runs the prediction, and sends the trajectory back.
4.  **Result Display:** The Hub marks the job as complete, and the web interface displays the 2D/3D trajectory and performance metrics.
5.  **Federated Learning:** In the background, the Node and Hub engage in an FL training round using the uploaded data to refine the global model.

## Security & Privacy

*   **Data Privacy:** Raw flight logs remain on the Worker Node during the training phase. Only model weights (gradients) are transmitted to the FL server.
*   **Internal Communication:** Node endpoints are designed for internal use by authorized hardware.
