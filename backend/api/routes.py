from fastapi import APIRouter, Response
from services.ai_detector import detect_macroplastic
from services.drift_engine import simulate_drift
from services.mission_planner import calculate_cleanup_mission
from services.mock_data import get_mock_aois, get_mock_dashboard_metrics

router = APIRouter(prefix="/api/v1")

@router.get("/aois")
async def list_aois():
    """
    Returns pre-staged Areas of Interest available for the demo (to populate UI config panels).
    """
    return get_mock_aois()

@router.get("/detect")
async def detect_plastic(aoi_id: str = "mumbai", s2_tile_path: str = None):
    """
    Returns the sub-pixel plastic detection polygons for a pre-staged Area of Interest (AOI).
    """
    return detect_macroplastic(aoi_id, s2_tile_path)

@router.get("/forecast")
async def forecast_drift(aoi_id: str = "mumbai", hours: int = 24):
    """
    Returns the predicted positions of the plastic patches over time based on wind and ocean currents.
    """
    if hours not in [24, 48, 72]:
        return {"error": "Invalid forecast step. Allowed values are [24, 48, 72]."}
    
    # 1. We optionally call detect_macroplastic to get the baseline detection (the "now" state)
    base_detect = detect_macroplastic(aoi_id)
    
    # 2. Pass those initial detections into the data science team's drift engine
    return simulate_drift(base_detect, aoi_id, hours)

@router.get("/mission")
async def plan_mission(aoi_id: str = "mumbai"):
    """
    Calculates the optimal cleanup route (TSP) over the top hotspots and returns the route.
    """
    # 1. We optionally call detect_macroplastic to get the baseline spots
    base_detect = detect_macroplastic(aoi_id)
    
    # 2. Pass those spots to the path optimisation engine
    return calculate_cleanup_mission(base_detect, aoi_id)

@router.get("/dashboard/metrics")
async def get_dashboard_stats(aoi_id: str = "mumbai"):
    """
    Returns aggregated figures (area covers, patch sizes, and biofouling degradation charts)
    for the side-panel elements described in the PRD.
    """
    return get_mock_dashboard_metrics(aoi_id)

@router.get("/mission/export")
async def export_mission_file(aoi_id: str = "mumbai", format: str = "gpx"):
    """
    Directly downloads a physical cleanup plan (GPX file) that a Coast Guard vessel 
    can load directly onto their nav-systems.
    """
    base_detect = detect_macroplastic(aoi_id)
    route = calculate_cleanup_mission(base_detect, aoi_id)
    
    # Grab the waypoints we planned
    coords = route["features"][0]["geometry"]["coordinates"]
    
    if format == "gpx":
        # Build raw GPX XML string manually for speed
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += '<gpx version="1.1" creator="DRIFT System">\n'
        xml_content += '  <trk>\n    <name>Coast Guard Cleanup Mission</name>\n    <trkseg>\n'
        
        for lon, lat in coords:
            xml_content += f'      <trkpt lat="{lat}" lon="{lon}"></trkpt>\n'
            
        xml_content += '    </trkseg>\n  </trk>\n</gpx>'
        
        # Stream the download back instantly as a file attachment
        return Response(
            content=xml_content,
            media_type="application/gpx+xml",
            headers={"Content-Disposition": f'attachment; filename="drift_mission_{aoi_id}.gpx"'}
        )
    else:
        # Default fallback to pure JSON output
        return route

