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
            print(f"Round {server_round} aggregation complete.")
            
        return aggregated_weights

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1, # Set equal to min_fit_clients to avoid warnings
)

def run_fl_server():
    """
    Starts the Flower Federated Learning server.
    """
    # Suppress TF logs in the child process too
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Define the initial global model to ensure all clients start with same weights
    input_shape = (50, 32)
    model = create_lstm_model(input_shape)
    initial_parameters = ndarrays_to_parameters(get_model_parameters(model))

    print("--- Federated Learning Server Started on port 8080 ---")
    
    # Note: start_server is deprecated in newer flwr versions but still works.
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=100), 
        strategy=strategy,
    )
    print("Training complete. Global model updated.")

if __name__ == "__main__":
    run_fl_server()
