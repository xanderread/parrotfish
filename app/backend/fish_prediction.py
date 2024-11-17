from fish_data_preprocess import fetch_latest_month
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from typing import List

# area to fish is the circle plot that trident will give you after running the algo
class FishingHotspot:
    def __init__(self, lat: float, long: float, tags: List[str]):
        self.lat = lat
        self.long = long
        self.tags = tags
        self.number_of_fish = 0.0

    def get_number_of_fish(self):
        return self.number_of_fish

    def get_tags(self):
        return self.tags

    def get_lat(self):
        return self.lat

    def get_long(self):
        return self.long

    
    def set_number_of_fish(self, fish: int):
        self.number_of_fish = fish

# Prepare Dataset
class FishDistributionDataset(Dataset):
    def __init__(self, data, input_seq_len, output_seq_len):
        self.data = data  # Shape: (time_steps, grid_size)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

    def __len__(self):
        return len(self.data) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.input_seq_len]  # Input sequence
        y = self.data[idx + self.input_seq_len: idx + self.input_seq_len + self.output_seq_len]  # Target
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  # Use the last output of the sequence
        return output

    # Make Predictions
    def predict(self, input_seq, days):
        self.eval()
        predictions = []
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(days):
                pred = self(input_seq)
                predictions.append(pred.numpy())
                input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
        return np.array(predictions).squeeze()



class FishPrediction():
    def __init__(self):
        # Region -> Population -> {model directory, data directory, bbox}
        self.models = {
            "Gulf of Mexico": {
                "Seabass": {
                    "model_dir": "data/gulf_of_mexico/seabass.pth",
                    "last_30_days_data": fetch_latest_month("data/gulf_of_mexico/seabass.csv",[-97.18345778791337, -82.32994217103956, 18.307385138607515, 31.029210975134845]), # (30 data points, grid_size)
                    "last_update": datetime(2024, 11, 14),
                },
                "bbox": [-97.18345778791337, -82.32994217103956, 18.307385138607515, 31.029210975134845] # [min_lon, max_lon, min_lat, max_lat]
            }
        }

    def which_region(self, start_lat, start_lon):
        """
            Determine the region based on the starting latitude and longitude.
            Return the region name.
        """
        for region, region_data in self.models.items():
            if start_lat >= region_data["bbox"][2] and start_lat <= region_data["bbox"][3] and start_lon >= region_data["bbox"][0] and start_lon <= region_data["bbox"][1]:
                return region
        return None
        
    
    def available_regions(self):
        return list(self.models.keys())
    
    def available_species(self, region):
        return list(self.models[region].keys())
    
    def predict(self, region, species, date):
        """
            Predict the fish distribution for a given region and species on a specific date.
            Return a grid of the predicted distribution along with the bounding box of this region [min lon, max lon, min lat, max lat].
        """
        # Check if region and species are available
        if region not in self.models:
            raise ValueError(f"Region '{region}' is not available.")
        if species not in self.models[region]:
            raise ValueError(f"Species '{species}' is not available in region '{region}'.")
        
        # Create a model using the interpolated data for the model size
        model = LSTMModel(input_size=self.models[region][species]["last_30_days_data"][0].shape[1], hidden_size=128, output_size=self.models[region][species]["last_30_days_data"][0].shape[1])
        model.load_state_dict(torch.load(self.models[region][species]["model_dir"]))

        # Determine number of days from last update to the requested date
        days = (date - self.models[region][species]["last_update"]).days

        print("days", days)

        # Predict the number of days since the input data
        predictions = model.predict(self.models[region][species]["last_30_days_data"][0], days)

        print("predictions", predictions.shape)

        # Check for a single day being predicted
        predicted_grids = [p.reshape(self.models[region][species]["last_30_days_data"][1]) for p in predictions]

        print("predicted_grids", predicted_grids)

        return predicted_grids[-1], self.models[region]["bbox"]

        # np.arange(solution[1][0], solution[1][1], 0.2), np.arange(solution[1][2], solution[1][3], 0.2) -> Converts bbox coords to grid points
        
    def bounds_to_coords(self, bounds):
        """
            Convert the bounding box to a list of coordinates.
        """
        return np.arange(bounds[0], bounds[1], 0.2), np.arange(bounds[2], bounds[3], 0.2)
    
    def prediction_to_hotspots(self, prediction, starting_lat, starting_lon, max_radius, species) -> List[FishingHotspot]:
        """
            Given a prediction, cut down to only cells outside of the max_radius from the starting_lat and starting_lon.
            Perform clustering on the remaining grid to find the hotspots.
            Return their coordinates and the number of fish in that hotspot (density cubed).

            Parameters: prediction: 2D numpy array of values
            Starting_lat: float
            Starting_lon: float
            Max_radius: float (km)
            species: string
        """
        x_coords, y_coords = self.bounds_to_coords(prediction[1])
        valid_points = []

        for i, lat in enumerate(y_coords):
            for j, lon in enumerate(x_coords):
                # Calculate distance from starting point
                distance = geopy.distance.distance(start_lat, start_lon, lat, lon).km
                
                if distance <= max_radius and prediction[i, j] > 0:
                    valid_points.append([lat, lon, prediction[i, j]])
        
        if valid_points == []:
            return []
        
        hotspots = []

        # Perform DBSCAN clustering
        # eps is in degrees (approximately 1km at the equator)
        eps = 0.009
        min_samples = 3
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(valid_points[:, :2])

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(valid_points[:, :2])
        labels = clustering.labels_

        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            # Get points in cluster
            cluster_mask = labels == label
            cluster_points = valid_points[cluster_mask]
            
            # Calculate cluster center (weighted by prediction values)
            weights = cluster_points[:, 2]
            center_lat = np.average(cluster_points[:, 0], weights=weights)
            center_lon = np.average(cluster_points[:, 1], weights=weights)
            
            # Calculate average intensity
            avg_intensity = np.mean(cluster_points[:, 2])
            
            hotspot = {
                'center_latitude': float(center_lat),
                'center_longitude': float(center_lon),
                'intensity': float(avg_intensity),
            }

            hostpot = FishingHotspot(
                lat=center_lat,
                long=center_lon,
                tags=[f"species:{species}"]
            )
            
            hostpot.set_number_of_fish(float(avg_intensity))

            hotspots.append(hotspot)

        return hotspots