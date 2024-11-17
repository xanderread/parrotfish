import torch
import torch.nn as nn
import math
from dataclasses import dataclass

import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import glob
import os

import pickle

from tqdm import tqdm

import matplotlib.pyplot as plt


@dataclass
class ModelConfig:
    """Store all hyperparameters and configuration"""

    input_size: int = 2
    hidden_size: int = 64
    num_layers: int = 1
    seq_length: int = 20
    pred_length: int = 10
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    loss_alpha: float = 0.25  # For combined loss function

    def save(self, path):
        """Save config to JSON"""
        if not path.endswith("/"):
            path += "/"
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(path + "traj_config.json", "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path):
        """Load config from JSON"""
        if not path.endswith("/"):
            path += "/"
        with open(path + "traj_config.json", "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class HaversineDistanceLoss(nn.Module):
    """
    Calculate the Haversine distance loss between predicted and actual coordinates
    This accounts for the spherical nature of Earth's coordinates
    """

    def __init__(self, radius=6371.0):  # Earth's radius in km
        super(HaversineDistanceLoss, self).__init__()
        self.radius = radius

    def forward(self, pred, target):
        # pred and target shape: (batch_size, seq_len, 2) where 2 is [lat, lon]

        # Convert to radians
        pred_lat = torch.deg2rad(pred[..., 0])
        pred_lon = torch.deg2rad(pred[..., 1])
        target_lat = torch.deg2rad(target[..., 0])
        target_lon = torch.deg2rad(target[..., 1])

        # Differences
        dlat = target_lat - pred_lat
        dlon = target_lon - pred_lon

        # Haversine formula
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = self.radius * c  # Distance in km

        return torch.mean(distance)


class CombinedDistanceLoss(nn.Module):
    """
    Combines MSE and Haversine distance loss
    """

    def __init__(self, alpha=0.25):
        super(CombinedDistanceLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.haversine_loss = HaversineDistanceLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        haversine = self.haversine_loss(pred, target)
        return self.alpha * mse + (1 - self.alpha) * haversine


# Modify the train_model function to use the new loss
def train_model(model, train_loader, valid_loader, config, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use the combined loss
    criterion = CombinedDistanceLoss(alpha=0.3)  # Adjust alpha as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        # Batch progress bar for training
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1} training", leave=False, unit="batch"
        )

        for batch_x, batch_y in train_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            # Convert loss to kilometers for display
            train_loss += loss.item() * batch_x.size(0)
            num_batches += batch_x.size(0)
            train_pbar.set_postfix({"distance_km": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        num_val_batches = 0

        valid_pbar = tqdm(
            valid_loader, desc=f"Epoch {epoch+1} validation", leave=False, unit="batch"
        )

        with torch.no_grad():
            for batch_x, batch_y in valid_pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                predictions, _ = model(batch_x)
                loss = criterion(predictions, batch_y)
                valid_loss += loss.item() * batch_x.size(0)
                num_val_batches += batch_x.size(0)
                valid_pbar.set_postfix({"distance_km": f"{loss.item():.4f}"})

        avg_valid_loss = valid_loss / num_val_batches
        valid_losses.append(avg_valid_loss)

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            {
                "train_dist": f"{avg_train_loss:.4f}km",
                "valid_dist": f"{avg_valid_loss:.4f}km",
            }
        )

    return train_losses, valid_losses


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs shape: (batch, seq_len, hidden_size)
        attention_weights = self.attention(encoder_outputs)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(
            attention_weights, dim=1
        )  # (batch, seq_len, 1)
        context = torch.bmm(
            attention_weights.transpose(1, 2), encoder_outputs
        )  # (batch, 1, hidden_size)
        return context.squeeze(1), attention_weights


class VesselTrajectoryPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_seq_len=1, dropout_rate=0.2):
        super(VesselTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len
        
        # Add dropout as a class parameter
        self.dropout = nn.Dropout(dropout_rate)
        
        # Add batch normalization layers
        self.encoder_norm = nn.BatchNorm1d(hidden_size * 2)  # *2 for bidirectional
        self.decoder_norm = nn.BatchNorm1d(hidden_size)
        
        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):        
        # Encode input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x)
        
        # Apply dropout to encoder outputs
        encoder_outputs = self.dropout(encoder_outputs)
        
        # Apply batch norm to encoder outputs
        # Reshape for batch norm
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (batch, hidden, seq_len)
        encoder_outputs = self.encoder_norm(encoder_outputs)
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (batch, seq_len, hidden)
        
        # Apply attention
        context, attention_weights = self.attention(encoder_outputs)
        
        # Apply dropout to context
        context = self.dropout(context)
        
        # Prepare decoder input
        decoder_input = context.unsqueeze(1).repeat(1, self.output_seq_len, 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        
        # Apply dropout to decoder output
        decoder_output = self.dropout(decoder_output)
        
        # Apply batch norm to decoder output
        decoder_output = decoder_output.transpose(1, 2)
        decoder_output = self.decoder_norm(decoder_output)
        decoder_output = decoder_output.transpose(1, 2)
        
        # Generate predictions
        predictions = self.output_layer(decoder_output)
        
        return predictions, attention_weights


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, seq_length, pred_length):
        """
        trajectories: list of numpy arrays, each array is a complete trajectory
        """
        self.sequence_starts = []  # Store all possible starting points
        self.trajectories = []  # Store corresponding trajectories
        self.seq_length = seq_length
        self.pred_length = pred_length

        for traj in trajectories:
            if len(traj) >= seq_length + pred_length:
                # For each trajectory, get all possible starting points
                for start_idx in range(len(traj) - (seq_length + pred_length) + 1):
                    self.sequence_starts.append(start_idx)
                    self.trajectories.append(traj)

    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        start_idx = self.sequence_starts[idx]

        # Get the sequence slice
        traj_slice = traj[start_idx : start_idx + self.seq_length + self.pred_length]

        # Split into input and target sequences
        x = traj_slice[: self.seq_length]
        y = traj_slice[self.seq_length : self.seq_length + self.pred_length]

        return torch.FloatTensor(x), torch.FloatTensor(y)
    
def prepare_trajectory_for_prediction(csv_path, ship_id=None):
    """
    Load trajectory data and prepare it for prediction.
    If ship_id is provided, return that specific trajectory,
    otherwise return a dictionary of all trajectories.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Sort by ship_id and timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['ship_id', 'timestamp'])
    
    # Group by ship_id and create trajectory arrays
    trajectories = {}
    for sid, group in df.groupby('ship_id'):
        # Convert to numpy array of [lat, lon] coordinates
        coords = group[['latitude', 'longitude']].values
        trajectories[sid] = coords
        
    if ship_id:
        if ship_id in trajectories:
            return trajectories[ship_id]
        else:
            raise ValueError(f"Ship ID {ship_id} not found in data")
    
    return trajectories

def predict_from_ais_data(model, scaler, csv_path, ship_id=None, num_predictions=10):
    """
    Load AIS data and make predictions
    
    Args:
        model: Trained trajectory prediction model
        scaler: Fitted scaler
        csv_path: Path to CSV file with AIS data
        ship_id: Optional specific ship to predict for
        num_predictions: Number of future points to predict
    
    Returns:
        Dictionary of predictions keyed by ship_id,
        or single prediction array if ship_id is provided
    """
    if ship_id:
        # Get single trajectory
        initial_sequence = prepare_trajectory_for_prediction(csv_path, ship_id)
        predictions = predict_trajectory(model, initial_sequence, num_predictions, scaler)
        return {ship_id: predictions}
    
    # Get all trajectories
    trajectories = prepare_trajectory_for_prediction(csv_path)
    predictions = {}
    
    for sid, trajectory in trajectories.items():
        try:
            pred = predict_trajectory(model, trajectory, num_predictions, scaler)
            predictions[sid] = pred
        except Exception as e:
            print(f"Failed to predict for ship {sid}: {e}")
            continue
    
    return predictions

def visualize_predictions(predictions, original_data, ship_id=None):
    """Visualize original trajectories and predictions"""
    plt.figure(figsize=(12, 8))
    
    if ship_id:
        ships_to_plot = [ship_id]
    else:
        ships_to_plot = list(predictions.keys())
    
    for sid in ships_to_plot:
        # Plot original trajectory
        orig = original_data[original_data['ship_id'] == sid]
        plt.plot(orig['longitude'], orig['latitude'], 'b.-', label=f'Original (Ship {sid})')
        
        # Plot prediction
        pred = predictions[sid]
        plt.plot(pred[:, 1], pred[:, 0], 'r.-', label=f'Predicted (Ship {sid})')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('AIS Trajectories: Original vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_model(model, scaler, config, path):
    """Save model and scaler"""
    if not path.endswith("/"):
        path += "/"
    os.makedirs(path, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), path + "model.pt")
    # Save scaler
    with open(path + "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    config.save(path)

    print(f"Model saved to {path}")
    print("Saved files:")
    print(f" - model.pt: Neural network weights")
    print(f" - scaler.pkl: Data normalization parameters")
    print(f" - config.json: Model configuration and hyperparameters")


def load_model(path):
    """Load model, scaler, and config"""
    if not path.endswith("/"):
        path += "/"

    # Load config
    config = ModelConfig.load(path)

    # Load model with proper configuration
    model = VesselTrajectoryPredictor(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_seq_len=config.pred_length,
        dropout_rate=0.2
    )
    model.load_state_dict(torch.load(path + "traj_model.pt"))

    # Load scaler
    with open(path + "traj_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, config


def evaluate_model(model, test_loader, scaler, device=None):
    """
    Evaluate model performance on test set
    Returns predictions, actual values, and various error metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    all_predictions = []
    all_targets = []
    total_mse = 0
    total_mae = 0
    total_samples = 0

    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Get predictions
            predictions, _ = model(batch_x)

            # Convert to numpy and inverse transform
            pred_np = predictions.cpu().numpy()
            target_np = batch_y.cpu().numpy()

            # Reshape for inverse transform if needed
            pred_shape = pred_np.shape
            target_shape = target_np.shape

            pred_np = pred_np.reshape(-1, 2)
            target_np = target_np.reshape(-1, 2)

            # Inverse transform
            pred_original = scaler.inverse_transform(pred_np)
            target_original = scaler.inverse_transform(target_np)

            # Reshape back
            pred_original = pred_original.reshape(pred_shape)
            target_original = target_original.reshape(target_shape)

            # Calculate errors
            mse = np.mean((pred_original - target_original) ** 2)
            mae = np.mean(np.abs(pred_original - target_original))

            # Accumulate metrics
            total_mse += mse * batch_x.size(0)
            total_mae += mae * batch_x.size(0)
            total_samples += batch_x.size(0)

            # Store predictions and targets
            all_predictions.append(pred_original)
            all_targets.append(target_original)

    # Calculate average metrics
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    rmse = np.sqrt(avg_mse)

    # Combine all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate distance errors
    distances = np.sqrt(np.sum((all_predictions - all_targets) ** 2, axis=-1))
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)

    # Print metrics
    print("\nTest Set Metrics:")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average Distance Error: {avg_distance:.6f}")
    print(f"Median Distance Error: {median_distance:.6f}")

    # Optional: Plot some sample predictions
    try:
        # Plot first 5 sequences
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(5, len(all_predictions))):
            pred = all_predictions[i]
            target = all_targets[i]

            axes[i].plot(pred[:, 0], pred[:, 1], "r-", label="Predicted")
            axes[i].plot(target[:, 0], target[:, 1], "b-", label="Actual")
            axes[i].set_title(f"Sequence {i+1}")
            axes[i].legend()

        plt.tight_layout()
        plt.savefig("test_predictions.png")
        print("\nSample predictions plot saved as 'test_predictions.png'")
    except Exception as e:
        print(f"Couldn't create plots: {e}")

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "mse": avg_mse,
        "rmse": rmse,
        "mae": avg_mae,
        "avg_distance": avg_distance,
        "median_distance": median_distance,
        "distances": distances,
    }


def load_trajectories_from_folder(folder_path):
    """Load all trajectory chunks and combine them"""
    trajectories = []
    for chunk_file in sorted(glob.glob(os.path.join(folder_path, "*.pkl"))):
        with open(chunk_file, "rb") as f:
            chunk_trajectories = pickle.load(f)
        trajectories.extend(chunk_trajectories)
    return trajectories


def prepare_data(folder_path, seq_length=20, pred_length=1, batch_size=32):
    """Load and prepare trajectory data"""
    # Load all trajectories
    print("Loading trajectories...")
    trajectories = []
    for chunk_file in sorted(glob.glob(os.path.join(folder_path, "*.pkl"))):
        with open(chunk_file, "rb") as f:
            chunk_trajectories = pickle.load(f)
            trajectories.extend(chunk_trajectories)  # Each element is a complete trajectory

    print(f"Loaded {len(trajectories)} trajectories")

    print(f"Total trajectories loaded: {len(trajectories)}")
    print(f"Sample trajectory lengths: {[len(t) for t in trajectories[:5]]}")

    # Normalize all trajectories using the same scaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()

    # Fit scaler on all points while keeping trajectories separate
    all_points = np.concatenate(trajectories)
    scaler.fit(all_points)

    # Scale each trajectory separately
    scaled_trajectories = [scaler.transform(traj) for traj in trajectories]

    # After scaling
    print("Sample scaled values:")
    print(scaled_trajectories[0][:5])  # First 5 points of first trajectory
    print("Original values:")
    print(trajectories[0][:5])

    # Split into train/valid/test
    num_trajectories = len(scaled_trajectories)
    train_size = int(num_trajectories * 0.7)
    valid_size = int(num_trajectories * 0.15)

    train_data = scaled_trajectories[:train_size]
    valid_data = scaled_trajectories[train_size : train_size + valid_size]
    test_data = scaled_trajectories[train_size + valid_size :]

    print(f"Train trajectories: {len(train_data)}")
    print(f"Valid trajectories: {len(valid_data)}")
    print(f"Test trajectories: {len(test_data)}")

    # Create datasets
    train_dataset = TrajectoryDataset(train_data, seq_length, pred_length)
    valid_dataset = TrajectoryDataset(valid_data, seq_length, pred_length)
    test_dataset = TrajectoryDataset(test_data, seq_length, pred_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Number of sequences in train dataset: {len(train_dataset)}")
    print(f"Number of sequences in valid dataset: {len(valid_dataset)}")
    print(f"Number of batches per epoch: {len(train_loader)}")

    return train_loader, valid_loader, test_loader, scaler


def predict_trajectory(model, initial_sequence, num_steps, scaler):
    """Make predictions using a trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Scale the input sequence using the same scaler used in training
    scaled_sequence = scaler.transform(initial_sequence)
    sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(device)

    predictions = []

    pbar = tqdm(range(num_steps), desc="Predicting", unit="step")

    with torch.no_grad():
        for _ in pbar:
            pred, _ = model(sequence_tensor)
            next_point = pred[:, -1:, :]

            # Add prediction to sequence and remove first point
            sequence_tensor = torch.cat([sequence_tensor[:, 1:, :], next_point], dim=1)

            # Store prediction
            predictions.append(next_point.cpu().numpy())

    # Combine predictions and inverse transform to get real coordinates
    predictions = np.concatenate(predictions, axis=1)
    predictions = scaler.inverse_transform(predictions.squeeze())

    return predictions


def start(trajectory_folder, config=None):
    """Train model with given or default configuration"""
    if config is None:
        config = ModelConfig()

    # Initialize model
    model = VesselTrajectoryPredictor(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_seq_len=config.pred_length,
    )

    # Prepare data
    train_loader, valid_loader, test_loader, scaler = prepare_data(
        trajectory_folder, config.seq_length, config.pred_length, config.batch_size
    )

    # Train model
    train_losses, valid_losses = train_model(
        model,
        train_loader,
        valid_loader,
        config=config,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
    )

    metrics = evaluate_model(model, test_loader, scaler)

    return model, scaler, config, (train_losses, valid_losses), metrics


if __name__ == "__main__":
    # # trajectory_folder = "/home/xander_read/kelechi-code/data"

    # trajectory_folder = "/Users/kelechi/Documents/Code/EF/data"

    # # Create custom config if needed
    # config = ModelConfig(
    #     seq_length=10, hidden_size=128, num_epochs=5, learning_rate=0.0005
    # )

    # # Train model
    # model, scaler, config, (train_losses, valid_losses), metrics = start(
    #     trajectory_folder, config=config
    # )

    # # Save everything
    # save_model(model, scaler, config, "models/")

    # print("\nTraining configuration:")
    # for key, value in config.__dict__.items():
    #     print(f"{key}: {value}")

    # print("\nTrain losses:", train_losses)
    # print("Validation losses:", valid_losses)

    # print("===============================")

    # print("\nDetailed test metrics:")
    # print(f"MSE: {metrics['mse']:.6f}")
    # print(f"RMSE: {metrics['rmse']:.6f}")
    # print(f"MAE: {metrics['mae']:.6f}")
    # print(f"Average Distance Error: {metrics['avg_distance']:.6f}")

    # Example of loading and using saved model
    loaded_model, loaded_scaler, loaded_config = load_model("../models/")
    print("\nLoaded model configuration:")
    for key, value in loaded_config.__dict__.items():
        print(f"{key}: {value}")

    predictions = predict_from_ais_data(
        loaded_model, 
        loaded_scaler, 
        '../data/ais/trajectories.csv',
        num_predictions=10
    )
    
    # Visualize
    df = pd.read_csv('../data/ais/trajectories.csv')
    visualize_predictions(predictions, df)
    # Or for single ship:
    # visualize_predictions(single_prediction, df, ship_id=ship_id)
    
    # Print prediction details
    for ship_id, pred in predictions.items():
        print(f"\nPredictions for Ship {ship_id}:")
        print("Original last position:", 
              df[df['ship_id'] == ship_id][['latitude', 'longitude']].values[-1])
        print("Predicted positions:")
        print(pred)
