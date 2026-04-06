import os
import pandas as pd
import io
import numpy as np
import json
import asyncio
from website_work.app.ml_models import preprocess_data, run_predictions, _load_drone_resources
from keras.models import load_model, Model
from typing import cast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEQ_LENGTH = 50
MODEL_FILENAME = "best_lstm_model.keras"


async def process_job(job_id, uav_model, csv_data):
    # 1. Prediction
    print(f"Running trajectory prediction... {job_id}")

    def _sync_process():
        df = pd.read_csv(io.StringIO(csv_data))
        preprocessed = preprocess_data(df.copy(), uav_model)
        df_clean = df.replace([float("inf"), float("-inf")], 0).ffill().bfill().fillna(0)
        return run_predictions(preprocessed, df_clean, uav_model)

    results = await asyncio.to_thread(_sync_process)
    return results


# =========================================================
# 2. PER-DRONE CONFIGURATION
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# app/ is current_dir. website_work/ is parent. root is grandparent.
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODELS_DIR = os.path.join(BASE_DIR, "website_work", "models")
# TRAINING_DATA_DIR = os.path.join(BASE_DIR, "20Jan") # If needed later

# Each drone maps to:
#   folder:  subfolder name in models/
#   targets: the 3 target column names for that drone
DRONE_CONFIG = {
    "DJI_Matrice_210": {
        "folder": "matrice_210",
        "targets": ["GPS:Long", "GPS:Lat", "GPS:heightMSL"],
    },
    "DJI_Matrice_600": {
        "folder": "matrice_600",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Mavic_2_Zoom": {
        "folder": "mavic_2_zoom",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Mavic_Pro": {
        "folder": "mavic_pro",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Phantom_4": {
        "folder": "Phantom_4",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
    "DJI_Phantom_4_Pro_V2": {
        "folder": "Phantom_4_Pro_V2",
        "targets": [
            "IMU_ATTI(0):Longitude",
            "IMU_ATTI(0):Latitude",
            "IMU_ATTI(0):alti:D",
        ],
    },
}


async def get_global_model(uav_model_name):
    config = DRONE_CONFIG.get(uav_model_name)
    if config is None:
        raise ValueError(
            f"Unknown UAV model: '{uav_model_name}'. "
            f"Valid options: {list(DRONE_CONFIG.keys())}"
        )

    folder = config["folder"]
    drone_models_dir = os.path.join(MODELS_DIR, folder)
    model_path = os.path.join(drone_models_dir, MODEL_FILENAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    global_model = await asyncio.to_thread(load_model, model_path)
    global_model = cast(Model, global_model)
    assert global_model is not None

    # Convert weights to list for JSON serialization
    weights = [w.tolist() for w in global_model.get_weights()]

    # Also return model architecture
    model_config = global_model.get_config()

    return {"weights": weights, "config": model_config}


async def get_processed_data(uav_model, flight_log):
    # Read CSV
    contents = await flight_log.read()

    def _sync_load_and_preprocess():
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        preprocessed_data = preprocess_data(df.copy(), uav_model)
        resources = _load_drone_resources(uav_model)
        return df, preprocessed_data, resources

    df, preprocessed_data, resources = await asyncio.to_thread(_sync_load_and_preprocess)

    input_scaler = resources["input_scaler"]
    target_scaler = resources["target_scaler"]
    target_cols = resources["target_cols"]

    # Prepare Input Features
    X_features = preprocessed_data.drop(columns=target_cols, errors="ignore").fillna(0)

    # Scale Inputs
    if input_scaler:
        X_scaled = input_scaler.transform(X_features)
    else:
        X_scaled = X_features.values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # Scale Targets
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0

    ground_truth_raw = df[target_cols].fillna(0).values
    if target_scaler:
        ground_truth_scaled = target_scaler.transform(ground_truth_raw)
    else:
        ground_truth_scaled = ground_truth_raw
    ground_truth_scaled = np.nan_to_num(ground_truth_scaled, nan=0.0)

    if len(X_scaled) < SEQ_LENGTH + 1:
        raise ValueError(f"Insufficient data for training. Need at least {SEQ_LENGTH + 1} rows.")

    # Create Sliding Windows
    X_sequences = []
    y_targets = []

    for i in range(len(X_scaled) - SEQ_LENGTH):
        X_sequences.append(X_scaled[i : i + SEQ_LENGTH])
        y_targets.append(ground_truth_scaled[i + SEQ_LENGTH])

    return {"x": np.array(X_sequences).tolist(), "y": np.array(y_targets).tolist()}


client_updates_dict = {}
MODEL_LIMIT = 1  # Threshold for aggregation
updates_lock = asyncio.Lock()


async def federated_average(uav_model, weights_json):
    global client_updates_dict

    # Validate UAV model early
    config = DRONE_CONFIG.get(uav_model)
    if config is None:
        raise ValueError(f"Unknown UAV model: {uav_model}")

    # Parse weights
    weights = json.loads(weights_json)

    avg_weights = None
    async with updates_lock:
        if uav_model not in client_updates_dict:
            client_updates_dict[uav_model] = []

        client_updates_dict[uav_model].append(weights)

        if len(client_updates_dict[uav_model]) >= MODEL_LIMIT:
            all_updates = client_updates_dict[uav_model]
            num_layers = len(all_updates[0])

            avg_weights = []
            for i in range(num_layers):
                layer_updates = [np.array(client[i]) for client in all_updates]
                avg_weights.append(np.mean(layer_updates, axis=0))

            # Clear queue
            client_updates_dict[uav_model] = []

    if avg_weights is not None:
        # Perform model update and save outside the lock
        folder = config["folder"]
        drone_models_dir = os.path.join(MODELS_DIR, folder)
        model_path = os.path.join(drone_models_dir, MODEL_FILENAME)

        def _sync_update_and_save():
            global_model = load_model(model_path)
            current_weights = global_model.get_weights()

            # Validation: ensure shapes match
            if len(avg_weights) != len(current_weights):
                raise ValueError(
                    f"Weight layer count mismatch. Expected {len(current_weights)}, got {len(avg_weights)}"
                )

            for i, (aw, cw) in enumerate(zip(avg_weights, current_weights)):
                if aw.shape != cw.shape:
                    raise ValueError(
                        f"Weight shape mismatch at layer {i}. Expected {cw.shape}, got {aw.shape}"
                    )

            global_model.set_weights(avg_weights)

            # Atomic save: write to temp, then rename
            temp_path = model_path + ".tmp"
            global_model.save(temp_path)
            os.replace(temp_path, model_path)

        await asyncio.to_thread(_sync_update_and_save)
        print(f"Federated Averaging completed for {uav_model}. Global model updated.")

