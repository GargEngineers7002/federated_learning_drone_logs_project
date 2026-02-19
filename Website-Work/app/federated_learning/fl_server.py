import flwr as fl
import os
import tensorflow as tf
from Website-Work.app.federated_learning.utils import create_lstm_model, set_model_parameters, get_model_parameters

# Strategy for aggregating weights
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Train on 100% of available clients
    min_fit_clients=2, # Minimum number of clients to start training
    min_available_clients=2,
)

def main():
    # Define the initial global model to ensure all clients start with same weights
    # Note: Input shape must match the client data (50 timesteps, 32 features for Matrice 210)
    input_shape = (50, 32) 
    model = create_lstm_model(input_shape)
    initial_parameters = fl.common.ndarrays_to_parameters(get_model_parameters(model))

    print("Starting Federated Learning Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # After training, you can save the final aggregated model
    # Note: In a production setting, you'd extract the weights from the strategy's final results
    print("Training complete. Global model updated.")

if __name__ == "__main__":
    main()
