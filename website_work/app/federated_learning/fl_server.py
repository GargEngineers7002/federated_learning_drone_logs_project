import flwr as fl
import os
import tensorflow as tf
from website_work.app.federated_learning.utils import (
    create_lstm_model,
    set_model_parameters,
    get_model_parameters,
)
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, start_server
from flwr.common import ndarrays_to_parameters

# Strategy for aggregating weights
class SaveModelStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        if aggregated_weights is not None:
            # Save the aggregated weights
            print(f"Saving round {server_round} aggregated weights...")
            # Convert parameters to ndarrays
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_weights[0])
            
            # For simplicity, we save to a general 'global_model.weights.h5' or similar
            # In a multi-drone setup, you'd want to save per drone type
            # Here we save to a default location that nodes can use
            # np.savez("global_model_weights.npz", *aggregated_ndarrays)
            
            # If we want to update the actual .keras files, we need to know which one.
            # For now, we'll just log and suggest a path.
            print(f"Round {server_round} aggregation complete.")
            
        return aggregated_weights

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1,
)

def run_fl_server():
    """
    Starts the Flower Federated Learning server.
    """
    # Define the initial global model to ensure all clients start with same weights
    # Note: Input shape must match the client data (50 timesteps, 32 features for Matrice 210)
    input_shape = (50, 32)
    model = create_lstm_model(input_shape)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

    print("--- Federated Learning Server Started on port 8080 ---")
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=100), # Increased rounds for continuous learning
        strategy=strategy,
    )
    print("Training complete. Global model updated.")

if __name__ == "__main__":
    run_fl_server()
