# trident algorithm 
from typing import List
from typing import NewType
from datetime import datetime  
from enum import Enum
import numpy as np
from fish_prediction import FishPrediction
import python_weather
from trajectory.backend_predict import predict
from weather import WeatherSailingScore
import asyncio
import os
import random
Meters = NewType('Meters', float)

HOTSPOT_RADIUS = Meters(1000)
CANDIDATE_HOTSPOTS = 10
MIN_DISTANCE_BETWEEN_HOTSPOTS = Meters(500000)

# Scoring constants
WEATHER_WEIGHT = 0 # for now
SHIP_WEIGHT = 0.5



fish_prediction = FishPrediction()
weather_sailing_score = WeatherSailingScore()

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


def get_number_of_ships_in_hotspot(hotspot: FishingHotspot, time_of_fishing: datetime) -> int:
    # call kelechi's ship api
    print("time of fishing", time_of_fishing)
    days_ahead = abs((time_of_fishing - datetime.now()).days)
    print("days ahead", days_ahead)
    predictions = predict(days_ahead)
    print("predictions", predictions)
    count = 0
    # check how many of the predictions are in the HOTSPOT_RADIUS
    for prediction in predictions:
        if haversine_distance(hotspot.get_lat(), hotspot.get_long(), prediction[0], prediction[1]) < HOTSPOT_RADIUS:
            count += 1
    if count == 0:
        print("no ships in hotspot")
        return 0
    print("got " + str(count) + " ships in hotspot")
    return count


def get_species_hotspots(start_lat: float, start_long: float, radius: float, species: str, date: datetime) -> List[FishingHotspot]:
    region = fish_prediction.which_region(start_lat, start_long)
    if region is None:
        print("No region found for this location")
        return []
    
    if species not in fish_prediction.available_species(region):
        print("No species found for this location")
        return []

    population_density, bbox = fish_prediction.predict(region, species, date)
    x_coords = np.arange(bbox[0], bbox[1], 0.2)
    y_coords = np.arange(bbox[2], bbox[3], 0.2)
    
    hotspots = []
    # Find local maxima in the population density
    height, width = population_density.shape
    for i in range(height):
        for j in range(width):
            if population_density[i][j] > 0:
                # Create potential new hotspot
                new_hotspot = FishingHotspot(
                    lat=y_coords[i],
                    long=x_coords[j],
                    tags=[f"species:{species}"]
                )
                new_hotspot.set_number_of_fish(float(population_density[i][j]))
                
                # Check distance from all existing hotspots
                too_close = False
                for existing_hotspot in hotspots:
                    dist = haversine_distance(
                        new_hotspot.get_lat(), new_hotspot.get_long(),
                        existing_hotspot.get_lat(), existing_hotspot.get_long()
                    )
                    if dist < MIN_DISTANCE_BETWEEN_HOTSPOTS:
                        too_close = True
                        break
                
                if not too_close:
                    hotspots.append(new_hotspot)
    
    print("got " + str(len(hotspots)) + " hotspots")
    return hotspots

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two points on Earth in meters."""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_lat/2) ** 2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * 
         np.sin(delta_lon/2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c



# return final_score, {
#             'score': final_score,
#             'detailed_scores': scores,
#             'conditions': weather_data
#         }

def get_hotspot_score(hotspot: FishingHotspot, time_of_fishing: datetime) -> float:
    weather_score_tuple = weather_sailing_score.get_weather_score(hotspot.get_lat(), hotspot.get_long(), time_of_fishing)
    weather_score = float(weather_score_tuple[0])
    ship_count = get_number_of_ships_in_hotspot(hotspot, time_of_fishing)
    ship_score = 1.0 / (ship_count + 1)
    
    return WEATHER_WEIGHT * weather_score + SHIP_WEIGHT * ship_score


def run(start_lat: float, start_long: float, max_distance: int, time_of_fishing: datetime, species: str) -> List[FishingHotspot]:
    print("the distance i get given is " + str(max_distance))
    # get the hotspots for a species
    species_hotspots = get_species_hotspots(start_lat, start_long, max_distance, species, time_of_fishing)

    # filter out hotspots that are too far away - not working because of distance shit fucked up

    # Create a deterministic seed based on input parameters
    # seed_string = f"{start_lat}{start_long}{max_distance}{time_of_fishing.timestamp()}{species}"
    # random.seed(hash(seed_string))

    # if len(species_hotspots) > CANDIDATE_HOTSPOTS:
    #     species_hotspots = random.sample(species_hotspots, CANDIDATE_HOTSPOTS)

    for hotspot in species_hotspots:
        print(hotspot.get_lat(), hotspot.get_long())
    
    # sort by combined score of weather and ships
    species_hotspots.sort(key=lambda x: get_hotspot_score(x, time_of_fishing), reverse=True)
    print("the n")
    return species_hotspots


def convert_meters_to_degrees(meters: Meters) -> float:
    # 1 degree of latitude = 111,139 meters
    return meters / 111139


class Location:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon 

    def get_lat(self) -> float:
        return self.lat

    def get_lon(self) -> float:
        return self.lon










