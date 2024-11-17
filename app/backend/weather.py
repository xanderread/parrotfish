from datetime import datetime
from typing import Dict, Optional, Tuple
import math
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

class WeatherSailingScore:
    def __init__(self):
        # Ideal conditions for sailing
        self.ideal_conditions = {
            'wind_speed': 15,  # knots
            'temperature': 22,  # Celsius
            'visibility': 1000,   # metres
            'wave_height': 1.0  # meters
        }
        
        # Weight factors for different parameters
        self.weights = {
            'wind_speed': 0.4,
            'wave_height': 0.2,
            'visibility': 0.1,
            'temperature': 0.1
        }

    def get_weather_score(self, lat: float, lon: float, forecast_date: datetime,
                         weather_data: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate sailing conditions score for given location and date.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            forecast_date (datetime): Date for forecast
            weather_data (Dict, optional): Weather data if already available
            
        Returns:
            Tuple[float, Dict]: Score between 0-1 and detailed conditions
        """
        # For a given lat, lon determine the timezone
        # Use timezone to get the forecast for the given date
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max", "wind_speed_10m_max"],
            "timezone": "auto",
            "start_date": forecast_date.strftime("%Y-%m-%d"),
            "end_date": forecast_date.strftime("%Y-%m-%d"),
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process the data for the location and date
        data = responses[0].Daily()

        daily_temperature_2m_max = data.Variables(0).ValuesAsNumpy()[0]
        daily_temperature_2m_min = data.Variables(1).ValuesAsNumpy()[0]
        daily_precipitation_probability_max = data.Variables(2).ValuesAsNumpy()[0]
        daily_wind_speed_10m_max = data.Variables(3).ValuesAsNumpy()[0]


        url = "https://marine-api.open-meteo.com/v1/marine"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "wave_height_max",
            "timezone": "auto",
            "start_date": forecast_date.strftime("%Y-%m-%d"),
            "end_date": forecast_date.strftime("%Y-%m-%d")
        }
        try:
            responses = openmeteo.weather_api(url, params=params)
            daily_wave_height_max = responses[0].Daily().Variables(0).ValuesAsNumpy()[0]
        except Exception as e:
            daily_wave_height_max = 0

        weather_data = {
            'wind_speed': daily_wind_speed_10m_max,        # knots
            'temperature': ((daily_temperature_2m_max - daily_temperature_2m_min)/2) + daily_temperature_2m_min,        # Celsius
            'visibility': 8,          # metres
            'precipitation_prob': daily_precipitation_probability_max,  # percentage
            'wave_height': daily_wave_height_max,       # meters
        }

        # Calculate individual scores
        scores = {
            'wind_speed': self._calculate_wind_score(weather_data['wind_speed']),
            'wave_height': self._calculate_wave_score(weather_data['wave_height']),
            'visibility': self._calculate_visibility_score(weather_data['visibility']),
            'temperature': self._calculate_temperature_score(weather_data['temperature'])
        }

        # Calculate weighted final score
        final_score = sum(scores[param] * self.weights[param] for param in scores)
        
        # Apply penalties for adverse conditions
        if weather_data["precipitation_prob"] > 0.7:
            final_score *= 0.7
        
        # Ensure score is between 0 and 1
        final_score = max(0, min(1, final_score))
        
        return final_score, {
            'score': final_score,
            'detailed_scores': scores,
            'conditions': weather_data
        }

    def _calculate_wind_score(self, wind_speed: float) -> float:
        """Score wind speed conditions."""
        if wind_speed < 5:  # Too little wind
            return 0.3
        elif wind_speed < 10:  # Light wind
            return 0.7
        elif wind_speed <= 20:  # Ideal range
            return 1.0
        elif wind_speed <= 25:  # Strong wind
            return 0.6
        else:  # Too windy
            return 0.2


    def _calculate_wave_score(self, wave_height: float) -> float:
        """Score wave height conditions."""
        if wave_height == 0:  # No wave height data available
            return -float('inf')
        elif wave_height < 0.5:
            return 1.0
        elif wave_height < 1.5:
            return 0.8
        elif wave_height < 2.5:
            return 0.5
        else:
            return 0.2

    def _calculate_visibility_score(self, visibility: float) -> float:
        """Score visibility conditions."""
        return min(1.0, visibility / self.ideal_conditions['visibility'])

    def _calculate_temperature_score(self, temperature: float) -> float:
        """Score temperature conditions."""
        diff = abs(temperature - self.ideal_conditions['temperature'])
        return max(0, 1 - (diff / 15))  # Linear decrease, 0 score at 15°C difference

# Example usage
if __name__ == "__main__":
    weather_scorer = WeatherSailingScore()
    
    # Example coordinates (San Francisco Bay)
    lat, lon = 37.8199, -122.4783
    forecast_date = datetime.now()
    
    # Get score
    score, details = weather_scorer.get_weather_score(lat, lon, forecast_date)
    
    print(f"Sailing Weather Score: {score:.2f}")
    print("\nDetailed Scores:")
    for param, score in details['detailed_scores'].items():
        print(f"{param}: {score:.2f}")