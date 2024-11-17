import anthropic
from datetime import datetime, timedelta
import json
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from system_prompt import system_prompt

@dataclass
class FishingPlan:
    location: Tuple[float, float]
    time: str
    distance: float
    species: List[str]

    def to_dict(self) -> dict:
        return {
            "location": list(self.location),
            "time": self.time,
            "distance": self.distance,
            "species": self.species
        }

class ParrotfishAI:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.geolocator = Nominatim(user_agent="parrotfish_ai")
        self.messages = []
        self.system_prompt = system_prompt

    def location_to_coordinates(self, location_string: str) -> Tuple[float, float]:
        """Convert a location string to latitude and longitude coordinates."""
        try:
            location = self.geolocator.geocode(location_string)
            if location:
                return (location.latitude, location.longitude)
            raise ValueError(f"Could not find coordinates for location: {location_string}")
        except GeocoderTimedOut:
            raise TimeoutError("Geocoding service timed out. Please try again.")

    def convert_to_km(self, value: float, unit: str) -> float:
        """Convert nautical miles or standard miles to kilometers."""
        conversion_rates = {
            "nautical": 1.852,  # 1 nautical mile = 1.852 km
            "miles": 1.60934    # 1 mile = 1.60934 km
        }
        if unit not in conversion_rates:
            raise ValueError(f"Unsupported unit: {unit}. Use 'nautical' or 'miles'")
        return value * conversion_rates[unit]

    def _prepare_tools(self) -> List[Dict]:
        """Prepare the tool definitions for the API call."""
        return [
            {
                "name": "location_to_coordinates",
                "description": "Convert a location string to latitude and longitude coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location_string": {
                            "type": "string",
                            "description": "The location to geocode (e.g., 'Boston Harbor')"
                        }
                    },
                    "required": ["location_string"]
                }
            },
            {
                "name": "convert_to_km",
                "description": "Convert distances from nautical miles or standard miles to kilometers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The distance value to convert"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["nautical", "miles"],
                            "description": "The unit to convert from ('nautical' or 'miles')"
                        }
                    },
                    "required": ["value", "unit"]
                }
            }
        ]

    def chat(self, user_input: str) -> str:
        """Process a user message and return the assistant's response."""
        # Add user message to conversation history
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        while True:
            try:
                # Make API call
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    system=self.system_prompt,
                    messages=self.messages,
                    tools=self._prepare_tools(),
                    max_tokens=2048,
                )

                # If Claude wants to use a tool
                if response.stop_reason == "tool_use":
                    # Get the tool call from the last content block
                    tool_calls = [content for content in response.content if content.type == "tool_use"]
                    if not tool_calls:
                        continue
                        
                    tool_use = tool_calls[0]  # Get the first tool call
                    print(f"======Claude wants to use the {tool_use.name} tool======")

                    # Process the tool call
                    try:
                        if tool_use.name == "location_to_coordinates":
                            result = self.location_to_coordinates(tool_use.input["location_string"])
                            tool_result = str(list(result))
                        elif tool_use.name == "convert_to_km":
                            result = self.convert_to_km(
                                float(tool_use.input["value"]),
                                tool_use.input["unit"]
                            )
                            tool_result = str(result)
                        else:
                            raise ValueError(f"Unknown tool: {tool_use.name}")

                    except Exception as e:
                        tool_result = str(e)

                    # Add tool use to messages
                    self.messages.append({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_use.id,
                                "name": tool_use.name,
                                "input": tool_use.input
                            }
                        ]
                    })

                    # Add tool result to messages
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": tool_result
                            }
                        ]
                    })

                    # Continue the loop to get Claude's next response
                    continue

                # Extract the final text response
                final_response = ""
                for content in response.content:
                    if content.type == "text":
                        final_response = content.text
                        break

                if not final_response:
                    final_response = "I apologize, but I couldn't generate a proper response."

                # Add assistant's response to conversation history
                self.messages.append({
                    "role": "assistant",
                    "content": final_response
                })

                return final_response

            except Exception as e:
                print(f"Debug - Exception details: {str(e)}")
                return f"An error occurred: {str(e)}"

    def extract_fishing_plan(self) -> Optional[FishingPlan]:
        """
        Extract the fishing plan from the conversation history if all required
        information has been gathered.
        """
        try:
            # Look for the extractedInformation tag in the last assistant message
            last_message = next(
                (msg for msg in reversed(self.messages) 
                 if msg["role"] == "assistant"),
                None
            )
            
            if not last_message:
                return None

            # Find content between <extractedInformation> tags
            content = last_message["content"]
            if isinstance(content, list):
                # If content is a list, look for text content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content = item.get("text", "")
                        break
                else:
                    return None

            start_tag = "<extractedInformation>"
            end_tag = "</extractedInformation>"
            
            if start_tag in content and end_tag in content:
                json_str = content.split(start_tag)[1].split(end_tag)[0]
                data = json.loads(json_str)
                
                return FishingPlan(
                    location=tuple(data["location"]),
                    time=data["time"],
                    distance=float(data["distance"]),
                    species=data["species"]
                )
            
            return None
            
        except Exception as e:
            print(f"Debug - Failed to extract fishing plan: {str(e)}")
            return None

def main():
    # Example usage
    api_key = "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"
    assistant = ParrotfishAI(api_key)
    
    print("Welcome to Parrotfish.ai!")
    print("AI: Welcome! I'd love to help plan your fishing expedition. To get started, please tell me about where you'd like to fish and how far you're willing to travel.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Thank you for using Parrotfish.ai!")
            break
            
        response = assistant.chat(user_input)
        print("AI:", response)
        
        # Check if we have a complete fishing plan
        plan = assistant.extract_fishing_plan()
        if plan:
            print("\nFishing Plan Completed!")
            print(json.dumps(plan.to_dict(), indent=2))


if __name__ == "__main__":
    main()