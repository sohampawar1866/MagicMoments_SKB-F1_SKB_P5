from typing import Dict, Any
from services.mock_data import get_mock_detection_geojson

def detect_macroplastic(aoi_id: str, s2_tile_path: str = None) -> Dict[str, Any]:
    """
    AI TEAM CONTRACT:
    =================
    Input:
      - aoi_id: An identifier for the area of interest (e.g., 'mumbai').
      - s2_tile_path: An optional local path to a pre-staged Sentinel-2 imagery tile (.SAFE or .tif).
      
    Task:
      Load the U-Net / Segformer model, perform inference on the 9-channel Sentinel imagery, 
      extract the sub-pixel plastic fraction, and return polygons of detected plastic.
      
    Output format:
      Must return a valid GeoJSON 'FeatureCollection' dictionary. Every feature MUST have 
      properties: ['id', 'confidence', 'area_sq_meters', 'age_days', 'type'].
      
    -----------------
    This currently returns mock data to keep the frontend running.
    Remove the mock return and replace it with your PyTorch inference later.
    """
    
    # TODO (AI Team):
    # 1. Load S2 tile using rasterio
    # 2. Extract specific bands + calculate FDI, NDVI indices
    # 3. Model inference: probs, fractions = model(features)
    # 4. Convert probabilty mask to GeoJSON polygons (using shapely / rasterio.features)
    
    # Fallback to mock data for MVP phase
    return get_mock_detection_geojson(aoi_id)
