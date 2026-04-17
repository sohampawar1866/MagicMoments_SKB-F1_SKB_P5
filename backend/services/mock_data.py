import random

def get_mock_detection_geojson(aoi_id: str):
    """
    Returns mock sub-pixel plastic detection polygons.
    """
    # Coordinates for Mumbai offshore area
    base_lon, base_lat = 72.8, 18.9
    
    features = []
    # Generate 5 random "plastic clusters"
    for i in range(5):
        lon = base_lon - random.uniform(0.1, 0.4)
        lat = base_lat + random.uniform(0.1, 0.4)
        # Small polygon representing some macroplastic
        polygon = [
            [lon, lat],
            [lon + 0.005, lat + 0.002],
            [lon + 0.004, lat - 0.003],
            [lon - 0.001, lat - 0.004],
            [lon, lat]
        ]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "id": f"patch_{i}",
                "confidence": round(random.uniform(0.5, 0.95), 2),
                "area_sq_meters": round(random.uniform(15, 100), 2),
                "age_days": random.randint(1, 40),
                "type": "macroplastic"
            }
        }
        features.append(feature)
        
    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_mock_forecast_geojson(aoi_id: str, hours: int):
    """
    Simulates where the plastic has drifted based on a time delta.
    """
    detect_base = get_mock_detection_geojson(aoi_id)
    
    # Simple fake drift: move east & south slightly based on hours
    drift_lon_per_hour = -0.001
    drift_lat_per_hour = -0.0005
    
    for f in detect_base["features"]:
        coords = f["geometry"]["coordinates"][0]
        drifted_coords = [
            [c[0] + (drift_lon_per_hour * hours), c[1] + (drift_lat_per_hour * hours)]
            for c in coords
        ]
        f["geometry"]["coordinates"] = [drifted_coords]
        f["properties"]["forecast_hour"] = hours
        
    return detect_base


def get_mock_mission_geojson(aoi_id: str):
    """
    Returns an optimal route connecting the top patches.
    """
    detect_base = get_mock_detection_geojson(aoi_id)
    
    # Just take the first coordinate of each feature to make a path
    waypoints = []
    for f in detect_base["features"][:4]:  # Top 4 targets
        center = f["geometry"]["coordinates"][0][0] # rough center
        waypoints.append(center)
        
    line_feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": waypoints
        },
        "properties": {
            "mission_id": "OP_CLEAN_ALPHA",
            "estimated_vessel_time_hours": round(random.uniform(2, 6), 1),
            "priority": "HIGH"
        }
    }
    
    return {
        "type": "FeatureCollection",
        "features": [line_feature]
    }
