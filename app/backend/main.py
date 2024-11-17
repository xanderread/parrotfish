from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uuid
from contextlib import asynccontextmanager
from trident import run
from datetime import datetime, timedelta


lat = 20
lon = -96
run(lat, lon, 10000, datetime.now(), "Seabass")


# Import your existing ParrotfishAI class
from agent import ParrotfishAI

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    session_id: str
    fishing_plan: Optional[dict] = None

# Add new Pydantic model for the trident request
class TridentRequest(BaseModel):
    lat: float
    lon: float
    max_distance: float
    species: str
    time_of_fishing: str

# Store active sessions
sessions: Dict[str, ParrotfishAI] = {}

# Cleanup manager for inactive sessions (you might want to add timeout logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup sessions when shutting down
    sessions.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "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"

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Get or create session
    session_id = request.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = ParrotfishAI(API_KEY)

    try:
        # Get the ParrotfishAI instance for this session
        parrotfish = sessions[session_id]
        
        # Process the message
        response = parrotfish.chat(request.message)
        
        # Try to extract fishing plan
        fishing_plan = parrotfish.extract_fishing_plan()
        fishing_plan_dict = fishing_plan.to_dict() if fishing_plan else None
        
        return ChatResponse(
            message=response,
            session_id=session_id,
            fishing_plan=fishing_plan_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

# Optional: Add endpoint to list active sessions
@app.get("/sessions")
async def list_sessions():
    return {"active_sessions": list(sessions.keys())}

@app.post("/trident")
async def run_trident(request: TridentRequest):
    print("I am running trident with the following parameters:")
    print(request)
    try:
        # Convert string to datetime object
        time_of_fishing = datetime.strptime(request.time_of_fishing, "%Y-%m-%d")
        
        # Convert distance to meters if it's in kilometers
        distance_in_meters = request.max_distance * 1000  # Assuming input is in kilometers
        
        # Run the trident algorithm with the datetime object
        hotspots = run(
            request.lat,
            request.lon,
            distance_in_meters,
            time_of_fishing,  # Pass the datetime object instead of string
            request.species.lower().title()  # Convert to lowercase then capitalize first letter
        )
        
        if not hotspots:
            return {"hotspots": []}
            
        hotspots_data = [
            {
                "lat": hotspot.get_lat(),
                "long": hotspot.get_long(),
                "fish_population_score": hotspot.get_number_of_fish() * 10000,        
            }
            for hotspot in hotspots
        ]
        
        return {"hotspots": hotspots_data}
        
    except Exception as e:
        print(f"Error in run_trident: {str(e)}")  # Add better error logging
        raise HTTPException(status_code=500, detail=str(e))