import asyncio
import sys
from pathlib import Path

# Fix relative imports
sys.path.append(str(Path(__file__).parent))

from backend.api.routes import (
    list_aois,
    detect_plastic,
    forecast_drift,
    plan_mission,
    get_dashboard_stats,
    get_environment_context,
    preview_deposition_alerts,
    export_mission_file
)

async def test_workflows():
    results = {}
    
    print("\n--- 1. Testing /aois ---")
    try:
        res = list_aois()
        print("Success! Found:", list(res.get("aois", {}).keys()))
        results["aois"] = "PASSED"
    except Exception as e:
        print("FAILED:", e)
        results["aois"] = f"FAILED: {e}"

    print("\n--- 2. Testing /detect ---")
    try:
        res = detect_plastic(aoi_id="mumbai")
        num_features = len(res.get("features", []))
        print(f"Success! Detected {num_features} plastic features.")
        results["detect"] = "PASSED"
    except Exception as e:
        print("FAILED:", e)
        results["detect"] = f"FAILED: {e}"

    print("\n--- 3. Testing /forecast ---")
    try:
        res = forecast_drift(aoi_id="mumbai", hours=24)
        print(f"Success! Generated forecast features.")
        results["forecast"] = "PASSED"
    except Exception as e:
        print("FAILED:", e)
        results["forecast"] = f"FAILED: {e}"

    print("\n--- 4. Testing /mission ---")
    try:
        res = plan_mission(aoi_id="mumbai")
        print(f"Success! Planned mission.")
        results["mission"] = "PASSED"
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("FAILED:", e)
        results["mission"] = f"FAILED: {e}"
        
    print("\n--- 5. Testing /dashboard/metrics ---")
    try:
        res = get_dashboard_stats(aoi_id="mumbai")
        print(f"Success! Got metrics.")
        results["metrics"] = "PASSED"
    except Exception as e:
        print("FAILED:", e)
        results["metrics"] = f"FAILED: {e}"

    print("\n--- 6. Testing /alerts/preview ---")
    try:
        res = preview_deposition_alerts(aoi_id="mumbai", hours=24)
        print(f"Success! Generated alerts.")
        results["alerts"] = "PASSED"
    except Exception as e:
        print("FAILED:", e)
        results["alerts"] = f"FAILED: {e}"

    print("\n\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("backend/.env")
    asyncio.run(test_workflows())
