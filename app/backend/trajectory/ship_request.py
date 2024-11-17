import asyncio
import websockets
import json
import datetime
from typing import Dict, List, Tuple

class AISStreamClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://stream.aisstream.io/v0/stream"
        self.vessels: Dict[str, Dict] = {}  # Store vessel positions by MMSI

    def create_subscription_message(self, bounds: Tuple[float, float, float, float]):
        """Create subscription message for the given bounding box
        bounds: (min_lat, min_lon, max_lat, max_lon)"""
        return {
            "APIKey": self.api_key,
            "BoundingBoxes": [[
                {"Latitude": bounds[0], "Longitude": bounds[1]},  # Southwest
                {"Latitude": bounds[2], "Longitude": bounds[3]}   # Northeast
            ]]
        }

    async def handle_message(self, message: Dict):
        """Process incoming AIS message"""
        try:
            message_type = message["MessageType"]
            
            if message_type == "PositionReport":
                ais_message = message["Message"]
                mmsi = str(ais_message["UserID"])
                
                self.vessels[mmsi] = {
                    "mmsi": mmsi,
                    "lat": ais_message["Latitude"],
                    "lon": ais_message["Longitude"],
                    "speed": ais_message.get("SpeedOverGround"),
                    "course": ais_message.get("CourseOverGround"),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }
        except Exception as e:
            print(f"Error processing message: {e}")

    async def connect_and_stream(self, bounds: Tuple[float, float, float, float], 
                               duration_seconds: int = 60):
        """Connect to AISStream and collect data for specified duration"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Subscribe to the stream
                subscription = self.create_subscription_message(bounds)
                await websocket.send(json.dumps(subscription))
                print(f"Subscribed to AISStream for region: {bounds}")

                # Set up timing
                start_time = datetime.datetime.utcnow()
                end_time = start_time + datetime.timedelta(seconds=duration_seconds)

                while datetime.datetime.utcnow() < end_time:
                    try:
                        message = await websocket.recv()
                        await self.handle_message(json.loads(message))
                        
                        # Print current vessel count periodically
                        if len(self.vessels) % 10 == 0:
                            print(f"Currently tracking {len(self.vessels)} vessels")
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed unexpectedly")
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

        except Exception as e:
            print(f"Connection error: {e}")
        
        return self.vessels

def get_vessels_in_region(api_key: str, 
                         bounds: Tuple[float, float, float, float],
                         duration_seconds: int = 60) -> Dict[str, Dict]:
    """
    Get vessel positions within a bounding box
    
    Args:
        api_key: AISStream API key
        bounds: (min_lat, min_lon, max_lat, max_lon)
        duration_seconds: How long to collect data
        
    Returns:
        Dictionary of vessel positions keyed by MMSI
    """
    client = AISStreamClient(api_key)
    return asyncio.run(client.connect_and_stream(bounds, duration_seconds))

# Example usage:
if __name__ == "__main__":
    API_KEY = "<>"
    
    # Example: Bounding box around part of the English Channel
    BOUNDS = (50.0, -4.0, 51.0, -3.0)  # (min_lat, min_lon, max_lat, max_lon)
    
    # Collect data for 5 minutes
    vessels = get_vessels_in_region(API_KEY, BOUNDS, duration_seconds=1000)
    
    # Print results
    print(f"\nFound {len(vessels)} vessels:")
    for mmsi, data in vessels.items():
        print(f"MMSI: {mmsi}")
        print(f"Position: {data['lat']:.4f}, {data['lon']:.4f}")
        print(f"Speed: {data['speed']} knots")
        print(f"Course: {data['course']}°")
        print(f"Last updated: {data['timestamp']}")
        print("---")