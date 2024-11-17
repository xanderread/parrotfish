# trident algorithm 
from typing import List
from typing import NewType
from datetime import datetime  
from enum import Enum
import numpy as np
from fish_prediction import FishPrediction
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
    return fish_prediction.prediction_to_hotspots(population_density, start_lat, start_long, radius, species)



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










