import pandas as pd
import numpy as np
import os
import pickle
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Standardize imports to project root
from website_work.app.preprocessing_scripts import (
    matrice_210,
    matrice_600,
    mavic_2_zoom,
    mavic_pro,
    phantom_4,
    phantom_4_pro_v2,
)

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

# MODEL FILE NAMES (same for every drone)
MODEL_FILES = {
    "LSTM": "best_lstm_model.keras",
    "GRU": "best_GRU_model.keras",
    "BiLSTM": "best_BiLSTM_model.keras",
    "RNN": "best_RNN_model.keras",
    "TCN": "best_TCN_model.keras",
}

# =========================================================
# 3. LAZY-LOADING CACHE
# =========================================================
_drone_cache = {}


def _load_drone_resources(uav_model_name: str) -> dict:
    """Load (or return cached) models + scalers for a specific drone."""
    config = DRONE_CONFIG.get(uav_model_name)
    if config is None:
        raise ValueError(
            f"Unknown UAV model: '{uav_model_name}'. "
            f"Valid options: {list(DRONE_CONFIG.keys())}"
        )

    folder = config["folder"]

    # Return from cache if already loaded
    if folder in _drone_cache:
        return _drone_cache[folder]

    print(f"[INFO] Loading resources for '{uav_model_name}' (folder: {folder})...")

    drone_models_dir = os.path.join(MODELS_DIR, folder)
    target_cols = config["targets"]

    # ------- Load Models -------
    loaded_models = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(drone_models_dir, filename)
        if os.path.exists(model_path):
            try:
                loaded_models[model_name] = load_model(model_path)
                print(f"  ✅ Loaded {model_name}: {filename}")
            except Exception as e:
                print(f"  ⚠️ Could not load {model_name}: {e}")
        else:
            print(f"  ⚠️ Model file not found: {model_path}")

    if not loaded_models:
        raise FileNotFoundError(
            f"No model files found in '{drone_models_dir}'. "
            f"Please place .keras files there."
        )

    # ------- Load Scalers -------
    input_scaler = None
    target_scaler = None
    input_columns = []

    scaler_path = os.path.join(drone_models_dir, "scaler.pkl")
    target_scaler_path = os.path.join(drone_models_dir, "target_scaler.pkl")
    try:
        if os.path.exists(scaler_path) and os.path.exists(target_scaler_path):
            with open(scaler_path, "rb") as f:
                input_scaler = pickle.load(f)
            with open(target_scaler_path, "rb") as f:
                target_scaler = pickle.load(f)
            print(f"  ✅ Loaded pre-trained scalers from pickles.")
    except Exception as e:
        print(f"  ⚠️ Could not load pickle scalers: {e}")

    # Cache everything
    entry = {
        "models": loaded_models,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "input_columns": input_columns,
        "target_cols": target_cols,
    }
    _drone_cache[folder] = entry
    print(f"[INFO] Resources for '{uav_model_name}' loaded and cached.\n")
    return entry


# =========================================================
# 4. PREPROCESSING ROUTER
# =========================================================
ROUTING_DICT = {
    "DJI_Matrice_210": matrice_210.preprocess,
    "DJI_Matrice_600": matrice_600.preprocess,
    "DJI_Mavic_2_Zoom": mavic_2_zoom.preprocess,
    "DJI_Mavic_Pro": mavic_pro.preprocess,
    "DJI_Phantom_4": phantom_4.preprocess,
    "DJI_Phantom_4_Pro_V2": phantom_4_pro_v2.preprocess,
}


def preprocess_data(dataframe, uav_model_name: str):
    func = ROUTING_DICT.get(uav_model_name)
    if func is None:
        raise ValueError(f"Unknown UAV model: '{uav_model_name}'")
    return func(dataframe)


# =========================================================
# 5. PREDICTION FUNCTION
# =========================================================
def run_predictions(preprocessed_data, original_df, uav_model_name: str):
    """Run all model predictions for the given drone type."""

    # Load the correct models & scalers for this drone
    resources = _load_drone_resources(uav_model_name)
    dl_models = resources["models"]
    input_scaler = resources["input_scaler"]
    target_scaler = resources["target_scaler"]
    input_columns = resources["input_columns"]
    target_cols = resources["target_cols"]

    # 1. Prepare Input Features
    X_features = preprocessed_data.drop(columns=target_cols, errors="ignore")

    # Fill NaNs
    X_features = X_features.fillna(0)

    # 2. Scale Inputs
    if input_scaler:
        try:
            X_scaled = input_scaler.transform(X_features)
            if np.isnan(X_scaled).any():
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        except Exception as e:
            print(f"Scaler Error: {e}. Using raw values.")
            X_scaled = X_features.values
    else:
        X_scaled = X_features.values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # 3. Create Sliding Windows
    SEQ_LENGTH = 50
    X_sequences = []
    for i in range(len(X_scaled) - SEQ_LENGTH + 1):
        X_sequences.append(X_scaled[i : i + SEQ_LENGTH])
    X_sequences = np.array(X_sequences)

    if len(X_sequences) == 0:
        return {}

    # 4. Prepare Ground Truth
    start_idx = SEQ_LENGTH
    end_idx = start_idx + len(X_sequences)

    try:
        ground_truth = original_df[target_cols].values[start_idx:end_idx]
        if np.isnan(ground_truth).any():
            ground_truth = np.nan_to_num(ground_truth, nan=0.0)
    except KeyError:
        ground_truth = None

    # Safety: ensure lengths match
    if ground_truth is not None:
        min_len = min(len(X_sequences), len(ground_truth))
        X_sequences = X_sequences[:min_len]
        ground_truth = ground_truth[:min_len]

    # 5. Run Each Model
    results = {}
    for name, model in dl_models.items():
        try:
            pred_scaled = model.predict(X_sequences, verbose=0)

            # Inverse scale
            if target_scaler:
                pred_real = target_scaler.inverse_transform(pred_scaled)
            else:
                pred_real = pred_scaled

            # Metrics
            if ground_truth is not None:
                rmse = np.sqrt(mean_squared_error(ground_truth, pred_real))
                mae = mean_absolute_error(ground_truth, pred_real)
                metrics = {"RMSE": f"{rmse:.4f}", "MAE": f"{mae:.4f}"}
            else:
                metrics = {"RMSE": "N/A", "MAE": "N/A"}

            results[name] = {
                "trajectory": {
                    "x": pred_real[:, 0].tolist(),
                    "y": pred_real[:, 1].tolist(),
                    "z": pred_real[:, 2].tolist(),
                },
                "metrics": metrics,
            }
        except Exception as e:
            print(f"  ❌ {name} Error: {e}")
            results[name] = {"error": str(e)}

    # 6. Add Actual Trajectory to response
    if ground_truth is not None and len(ground_truth) > 0:
        results["actual_trajectory"] = {
            "x": ground_truth[:, 0].tolist(),
            "y": ground_truth[:, 1].tolist(),
            "z": ground_truth[:, 2].tolist(),
        }

    return results
