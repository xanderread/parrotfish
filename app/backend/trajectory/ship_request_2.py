import asyncio
import websockets
import json
from datetime import datetime, timezone
from collections import defaultdict
import pickle
import pandas as pd
import os

async def connect_ais_stream(duration_seconds=2):
   # Dictionary to store trajectories by ship ID
   ship_trajectories = defaultdict(list)
   
   # Start time
   start_time = datetime.now(timezone.utc)
   
   async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
       subscribe_message = {
           "APIKey": "3e18cf7ad5ec1a8f8ba0d24508eeada39492d6b0", 
           "BoundingBoxes": [[[-97.19, -82.33, 18.31, 31.03]]]
       }
       subscribe_message_json = json.dumps(subscribe_message)
       await websocket.send(subscribe_message_json)
       
       try:
           async for message_json in websocket:
               # Check if duration has elapsed
               if (datetime.now(timezone.utc) - start_time).total_seconds() > duration_seconds:
                   break
                   
               message = json.loads(message_json)
               message_type = message["MessageType"]
               
               if message_type == "PositionReport":
                   ais_message = message['Message']['PositionReport']
                   ship_id = ais_message['UserID']
                   
                   # Store position with timestamp
                   position = {
                       'ship_id': ship_id,
                       'latitude': ais_message['Latitude'],
                       'longitude': ais_message['Longitude'],
                       'timestamp': datetime.now(timezone.utc).isoformat()
                   }
                   
                   ship_trajectories[ship_id].append(position)
                   
                   print(f"[{datetime.now(timezone.utc)}] ShipId: {ship_id} "
                         f"Latitude: {position['latitude']} "
                         f"Longitude: {position['longitude']}")
                   
       except Exception as e:
           print(f"Error: {e}")
   
   # Filter for ships with multiple positions
   multi_position_trajectories = {
       ship_id: positions 
       for ship_id, positions in ship_trajectories.items() 
       if len(positions) > 1
   }
   
   # Print summary
   print("\nTrajectory Summary:")
   print(f"Total ships detected: {len(ship_trajectories)}")
   print(f"Ships with multiple positions: {len(multi_position_trajectories)}")
   for ship_id, positions in multi_position_trajectories.items():
       print(f"Ship {ship_id}: {len(positions)} positions")
   
   # Create output directory if it doesn't exist
   os.makedirs('ais_data', exist_ok=True)
   
   # Save as pickle
   pickle_path = '../data/ais/trajectories.pkl'
   with open(pickle_path, 'wb') as f:
       pickle.dump(multi_position_trajectories, f)
   print(f"\nTrajectories saved to '{pickle_path}'")
   
   # Save as CSV
   # Convert to DataFrame
   all_positions = []
   for ship_id, positions in multi_position_trajectories.items():
       all_positions.extend(positions)
   
   df = pd.DataFrame(all_positions)
   csv_path = '../data/ais/trajectories.csv'
   df.to_csv(csv_path, index=False)
   print(f"Trajectories saved to '{csv_path}'")
   
   return multi_position_trajectories

def load_trajectories(format='pickle'):
   """Helper function to load saved trajectories"""
   if format == 'pickle':
       with open('../data/ais/trajectories.pkl', 'rb') as f:
           return pickle.load(f)
   elif format == 'csv':
       df = pd.read_csv('../data/ais/trajectories.csv')
       # Convert back to dictionary format
       trajectories = defaultdict(list)
       for _, row in df.iterrows():
           position = row.to_dict()
           trajectories[position['ship_id']].append(position)
       return dict(trajectories)
   else:
       raise ValueError("Format must be 'pickle' or 'csv'")

if __name__ == "__main__":
   # Run for 2 seconds
   trajectories = asyncio.run(connect_ais_stream(duration_seconds=60))