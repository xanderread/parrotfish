from .runner import predict_from_ais_data, load_model
import os

def predict(steps):
    # Get the absolute path to the backend directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    
    # Construct absolute paths
    models_path = os.path.join(backend_dir, "models")
    data_path = os.path.join(backend_dir, "data", "ais", "trajectories.csv")
    print(models_path)
    print(data_path)
    loaded_model, loaded_scaler, loaded_config = load_model(models_path)
    predictions = predict_from_ais_data(loaded_model, loaded_scaler, data_path, num_predictions=steps)
    
    print(predictions)
    return predictions

if __name__ == '__main__':
    steps = 5
    predict(steps)