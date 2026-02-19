import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from website_work.app.ml_models import DRONE_CONFIG, preprocess_data


def create_lstm_model(input_shape, num_targets=3):
    """Recreates the LSTM architecture found in the project."""
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(60, activation="relu"),
            BatchNormalization(),
            Dense(num_targets),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def get_model_parameters(model):
    return model.get_weights()


def set_model_parameters(model, parameters):
    model.set_weights(parameters)


def save_model_weights(model, weights, save_path):
    """Updates a model with new weights and saves it to a .keras file."""
    model.set_weights(weights)
    model.save(save_path)
    print(f"Model saved to {save_path}")


def load_local_data(csv_path, uav_model_name, seq_length=50):
    """Loads and preprocesses local data for training."""
    df = pd.read_csv(csv_path)
    preprocessed = preprocess_data(df.copy(), uav_model_name)

    # Fill NaNs
    preprocessed = preprocessed.fillna(0)

    # Simple scaling for demonstration (ideally use your saved scalers)
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(preprocessed)

    X, y = [], []
    # Assuming the first 3 columns are the targets based on DRONE_CONFIG
    # In a real scenario, we'd match column names exactly
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i : i + seq_length])
        y.append(scaled_data[i + seq_length, :3])

    return np.array(X), np.array(y)
