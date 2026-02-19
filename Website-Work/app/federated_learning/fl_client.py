import os
import requests
import pandas as pd
import flwr as fl
import tensorflow as tf
from Website-Work.app.ml_models import preprocess_data, run_predictions
from Website-Work.app.federated_learning.utils import create_lstm_model, load_local_data, set_model_parameters, get_model_parameters

class DroneClient(fl.client.NumPyClient):
    def __init__(self, uav_model, csv_path, server_url):
        self.uav_model = uav_model
        self.csv_path = csv_path
        self.server_url = server_url
        
        # Load local data for training
        self.x_train, self.y_train = load_local_data(csv_path, uav_model)
        
        # Initialize model
        input_shape = (self.x_train.shape[1], self.x_train.shape[2])
        self.model = create_lstm_model(input_shape)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return get_model_parameters(self.model), len(self.x_train), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, mae = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss, len(self.x_train), {"mae": mae}

def run_prediction_and_upload(uav_model, csv_path, upload_url):
    """
    Runs local predictions and uploads them to the central website.
    """
    df = pd.read_csv(csv_path)
    preprocessed = preprocess_data(df.copy(), uav_model)
    
    # Clean raw data as done in main.py
    df_clean = df.replace([float('inf'), float('-inf')], 0).ffill().bfill().fillna(0)
    
    results = run_predictions(preprocessed, df_clean, uav_model)
    
    # Upload results to the central server
    payload = {
        "uav_model": uav_model,
        "results": results
    }
    try:
        response = requests.post(upload_url, json=payload)
        print(f"Upload Status: {response.status_code}")
    except Exception as e:
        print(f"Failed to upload results: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flower Client for UAV Trajectory Prediction")
    parser.add_argument(
        "--uav_type",
        type=str,
        default="DJI_Matrice_210",
        help="Type of UAV model (e.g., DJI_Matrice_210)",
    )
    parser.add_argument(
        "--csv_log",
        type=str,
        default="Website-Work/FLY007.csv",
        help="Path to the local CSV flight log file",
    )
    parser.add_argument(
        "--website_api_url",
        type=str,
        default="http://localhost:8000/api/upload_results",
        help="URL of the central website API for uploading prediction results",
    )
    parser.add_argument(
        "--fl_server_address",
        type=str,
        default="localhost:8080",
        help="Address of the Federated Learning server (e.g., localhost:8080)",
    )
    args = parser.parse_args()

    # 1. Run local prediction and upload to website
    print(f"Running prediction for {args.uav_type}...")
    run_prediction_and_upload(args.uav_type, args.csv_log, args.website_api_url)

    # 2. Start FL training
    print("Starting Federated Learning training round...")
    fl.client.start_numpy_client(
        server_address=args.fl_server_address,
        client=DroneClient(args.uav_type, args.csv_log, args.website_api_url)
    )
