from fastapi import APIRouter
from services.mock_data import (
    get_mock_detection_geojson,
    get_mock_forecast_geojson,
    get_mock_mission_geojson
)

router = APIRouter(prefix="/api/v1")

@router.get("/detect")
async def detect_plastic(aoi_id: str = "mumbai"):
    """
    Returns the sub-pixel plastic detection polygons for a pre-staged Area of Interest (AOI).
    """
    return get_mock_detection_geojson(aoi_id)

@router.get("/forecast")
async def forecast_drift(aoi_id: str = "mumbai", hours: int = 24):
    """
    Returns the predicted positions of the plastic patches over time based on wind and ocean currents.
    """
    if hours not in [24, 48, 72]:
        return {"error": "Invalid forecast step. Allowed values are [24, 48, 72]."}
    
    return get_mock_forecast_geojson(aoi_id, hours)

@router.get("/mission")
async def plan_mission(aoi_id: str = "mumbai"):
    """
    Calculates the optimal cleanup route (TSP) over the top hotspots and returns the route.
    """
    return get_mock_mission_geojson(aoi_id)
