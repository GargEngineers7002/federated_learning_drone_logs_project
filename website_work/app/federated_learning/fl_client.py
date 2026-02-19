import os
import requests
import pandas as pd
import io
import time
import numpy as np
from flwr.client import NumPyClient, start_numpy_client
import tensorflow as tf
from website_work.app.ml_models import preprocess_data, run_predictions
from website_work.app.federated_learning.utils import (
    create_lstm_model,
    load_local_data,
    set_model_parameters,
    get_model_parameters,
)


class DroneClient(NumPyClient):
    def __init__(self, uav_model, df):
        self.uav_model = uav_model
        
        # Preprocess the data for training
        # Note: In a real scenario, we'd use more sophisticated training data management
        # For now, we use the same data that was uploaded for prediction
        preprocessed = preprocess_data(df.copy(), uav_model)
        preprocessed = preprocessed.fillna(0)
        
        # Prepare sequences (assuming 50 timesteps)
        X, y = [], []
        seq_length = 50
        scaled_data = preprocessed.values # raw for demonstration
        
        if len(scaled_data) > seq_length:
            for i in range(len(scaled_data) - seq_length):
                X.append(scaled_data[i : i + seq_length])
                y.append(scaled_data[i + seq_length, :3])
            self.x_train = np.array(X)
            self.y_train = np.array(y)
        else:
            # Fallback if data is too short
            self.x_train = np.zeros((1, 50, preprocessed.shape[1]))
            self.y_train = np.zeros((1, 3))

        # Initialize model
        input_shape = (self.x_train.shape[1], self.x_train.shape[2])
        self.model = create_lstm_model(input_shape)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        # Train for 1 epoch on the received data
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return get_model_parameters(self.model), len(self.x_train), {"uav_model": self.uav_model}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"mae": mae}


def process_job(job_id, uav_model, csv_data, website_api_url, fl_server_address):
    """
    1. Runs local prediction.
    2. Uploads results to central server.
    3. Participates in FL round.
    """
    print(f"[*] Processing job {job_id} for {uav_model}...")
    
    # 1. Prediction
    df = pd.read_csv(io.StringIO(csv_data))
    preprocessed = preprocess_data(df.copy(), uav_model)
    
    # Clean raw data
    df_clean = df.replace([float("inf"), float("-inf")], 0).ffill().bfill().fillna(0)
    
    results = run_predictions(preprocessed, df_clean, uav_model)
    
    # 2. Upload Prediction Results
    submit_url = f"{website_api_url}/api/node/submit_results"
    payload = {"job_id": job_id, "results": results}
    try:
        resp = requests.post(submit_url, json=payload)
        print(f"    [+] Prediction results submitted: {resp.status_code}")
    except Exception as e:
        print(f"    [!] Failed to submit results: {e}")

    # 3. FL Training Round
    print(f"    [*] Starting FL training round for job {job_id}...")
    try:
        # Start a one-round client
        start_numpy_client(
            server_address=fl_server_address,
            client=DroneClient(uav_model, df),
        )
        print(f"    [+] FL round completed.")
    except Exception as e:
        print(f"    [!] FL training failed: {e}")


def worker_loop(website_api_url, fl_server_address):
    print(f"[*] Node worker started. Polling {website_api_url}...")
    while True:
        try:
            resp = requests.get(f"{website_api_url}/api/node/get_job")
            if resp.status_code == 200:
                job_data = resp.json()
                if job_data.get("job_id"):
                    process_job(
                        job_data["job_id"],
                        job_data["uav_model"],
                        job_data["data"],
                        website_api_url,
                        fl_server_address
                    )
                else:
                    # No jobs, wait a bit
                    time.sleep(2)
            else:
                print(f"[!] Server error: {resp.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"[!] Connection error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Worker Node for Federated Learning")
    parser.add_argument(
        "--website_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the central website API",
    )
    parser.add_argument(
        "--fl_server",
        type=str,
        default="localhost:8080",
        help="Address of the Federated Learning server",
    )
    args = parser.parse_args()

    worker_loop(args.website_url, args.fl_server)
