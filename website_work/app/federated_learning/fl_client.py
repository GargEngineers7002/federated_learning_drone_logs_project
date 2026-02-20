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
        print(f"    [NODE-FL] Initializing training client for {uav_model}...")
        
        preprocessed = preprocess_data(df.copy(), uav_model)
        preprocessed = preprocessed.fillna(0)
        
        X, y = [], []
        seq_length = 50
        scaled_data = preprocessed.values 
        
        if len(scaled_data) > seq_length:
            for i in range(len(scaled_data) - seq_length):
                X.append(scaled_data[i : i + seq_length])
                y.append(scaled_data[i + seq_length, :3])
            self.x_train = np.array(X)
            self.y_train = np.array(y)
        else:
            self.x_train = np.zeros((1, 50, preprocessed.shape[1]))
            self.y_train = np.zeros((1, 3))

        input_shape = (self.x_train.shape[1], self.x_train.shape[2])
        self.model = create_lstm_model(input_shape)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        print(f"    [NODE-FL] Round started. Training on local data...")
        set_model_parameters(self.model, parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        print(f"    [NODE-FL] Training finished. Sending updated weights to server.")
        return get_model_parameters(self.model), len(self.x_train), {"uav_model": self.uav_model}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print(f"    [NODE-FL] Evaluation: Loss={loss:.4f}, MAE={mae:.4f}")
        return loss, len(self.x_train), {"mae": mae}


def process_job(job_id, uav_model, csv_data, website_api_url, fl_server_address):
    print("\n" + "-"*40)
    print(f"🛠️  PROCESSING JOB: {job_id}")
    print(f"📦 UAV Model: {uav_model}")
    print("-"*40)
    
    # 1. Prediction
    print("[1/3] Running trajectory prediction...")
    df = pd.read_csv(io.StringIO(csv_data))
    preprocessed = preprocess_data(df.copy(), uav_model)
    df_clean = df.replace([float("inf"), float("-inf")], 0).ffill().bfill().fillna(0)
    
    results = run_predictions(preprocessed, df_clean, uav_model)
    
    # 2. Upload Prediction Results
    print("[2/3] Submitting results to Central Hub...")
    submit_url = f"{website_api_url}/api/node/submit_results"
    payload = {"job_id": job_id, "results": results}
    try:
        resp = requests.post(submit_url, json=payload)
        if resp.status_code == 200:
            print(f"✅ SUCCESS: Results accepted by Hub.")
        else:
            print(f"❌ ERROR: Hub rejected results (Status: {resp.status_code})")
    except Exception as e:
        print(f"❌ ERROR: Connection failed during result submission: {e}")

    # 3. FL Training Round
    print("[3/3] Joining Federated Learning training round...")
    try:
        start_numpy_client(
            server_address=fl_server_address,
            client=DroneClient(uav_model, df),
        )
        print(f"✅ SUCCESS: FL training round finished.")
    except Exception as e:
        print(f"⚠️  WARNING: FL training failed or timed out: {e}")
    
    print("-"*40)
    print("✨ Job Complete. Returning to standby.")


def worker_loop(website_api_url, fl_server_address):
    print("\n" + "="*50)
    print("🛰️  WORKER NODE INITIALIZED")
    print(f"📍 Hub API: {website_api_url}")
    print(f"📍 FL Server: {fl_server_address}")
    print("="*50)
    print("[*] Polling for jobs...")
    
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
                    print("\n[*] Polling for new jobs...")
                else:
                    # No jobs, silent wait
                    time.sleep(2)
            else:
                print(f"[!] Server returned error: {resp.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"[!] Connection to Hub failed: {e}")
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
