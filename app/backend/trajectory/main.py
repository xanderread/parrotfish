import geopandas as gpd
import pandas as pd
import numpy as np
import psutil
import pickle

def extract_coordinates_from_multilinestring(geom):
    # Get coordinates from all line segments
    all_coords = []
    for line in geom.geoms:  # geoms gives us access to each line in the multilinestring
        coords = np.array(line.coords)
        all_coords.extend(coords)
    return np.array(all_coords)

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def calculate_velocity(coords):
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    times = np.ones_like(distances)  # If you have timestamps
    return distances/times
    

def process_gpkg(filepath, rows_per_chunk=1024):
    # Add trajectory filtering
    min_points = 10  # Minimum trajectory length
    max_points = 1000  # Maximum trajectory length

    print("Starting processing...")
    print_memory_usage()
    chunk_number = 0
    total_trajectories = 0
    
    try:
        while True:
            chunk = slice(chunk_number * rows_per_chunk, (chunk_number + 1) * rows_per_chunk)
            data_chunk = gpd.read_file(filepath, columns=[], rows=chunk)
            
            if data_chunk.shape[0] == 0:
                break
            
            chunk_trajectories = []
            
            print(f"\nProcessing chunk {chunk_number}")
            print(f"Number of trajectories in this chunk: {len(data_chunk)}")
            
            for index, row in data_chunk.iterrows():
                geom = row.geometry
                coords = extract_coordinates_from_multilinestring(geom)
                chunk_trajectories.append(coords.tolist())

                # velocities = calculate_velocity(coords)
                # if len(coords) >= min_points and len(coords) <= max_points:
                #     if np.all(velocities < max_velocity):
                #         chunk_trajectories.append(coords.tolist())
                
                if index == 0 and chunk_number == 0:  # Print info about first trajectory
                    print(f"\nSample trajectory shape: {coords.shape}")
                    print("First few points of first trajectory:")
                    print(coords[:5])
            
            if chunk_trajectories:
                output_file = f'./csvs/trajectory_chunk_{chunk_number}.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(chunk_trajectories, f)
                total_trajectories += len(chunk_trajectories)
                print(f"Saved chunk {chunk_number} with {len(chunk_trajectories)} trajectories")
            
            chunk_number += 1
            del data_chunk
            del chunk_trajectories
            
    except Exception as e:
        print(f"Error processing file: {e}")
        raise e
        
    print(f"\nFinished processing {total_trajectories} trajectories in {chunk_number} chunks")
    return chunk_number, total_trajectories

if __name__ == "__main__":
    # filepath = "/mnt/disks/data-dir/AISVesselTracks2023.gpkg"
    filepath = "~/Documents/EF-HACK/AISVesselTracks2023.gpkg"
    num_chunks, num_trajectories = process_gpkg(filepath)