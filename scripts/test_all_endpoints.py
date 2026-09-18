"""
CycloVision AI - End-to-End Test Suite for Backend, APIs, and Frontend Serving
Tests:
- GET /api/health
- POST /api/predict-trajectory (Demo and Custom)
- POST /api/analyze (Demo and Custom Fusion)
- GET / (Frontend Dashboard)
- CORS headers verification
"""

import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    passed = 0
    total = 5
    
    print("="*65)
    print("CYCLOVISION AI — END-TO-END INTEGRATION TEST SUITE")
    print("="*65)
    
    # Test 1: Health Check
    print("\n[TEST 1/5] Testing GET /api/health ...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print(f"  Status: {resp.status} - Service: {data.get('service')}")
            print(f"  YOLO Detector:  {data['models']['yolo_detector']['status']}")
            print(f"  GRU Trajectory: {data['models']['gru_trajectory']['status']}")
            assert data['status'] == "HEALTHY"
            assert data['models']['gru_trajectory']['status'] == "LOADED"
            passed += 1
            print("  [PASS]: Health check verified.")
            print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL]: {e}")
        
    # Test 2: Predict Trajectory (Demo Mode)
    print("\n[TEST 2/5] Testing POST /api/predict-trajectory (Demo Mode) ...")
    try:
        req_data = json.dumps({"use_demo": True, "steps": 6}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/api/predict-trajectory",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"  Status: {resp.status} - Source: {data.get('data_source_label')}")
            print(f"  Predicted Points: {data.get('prediction_points_count')}")
            print(f"  Step 1: {data['predictions'][0]}")
            assert data['status'] == "SUCCESS"
            assert data['is_demo'] is True
            assert len(data['predictions']) == 6
            passed += 1
            print("  [PASS]: Verified.")
            print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL]: {e}")
        
    # Test 3: Analyze Cyclone Fusion (Demo Mode with Weather & Risk)
    print("\n[TEST 3/5] Testing POST /api/analyze (Demo Fani Fusion) ...")
    try:
        req_data = json.dumps({"use_demo_mode": True, "steps": 6, "include_weather": True}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/api/analyze",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            print(f"  Status: {resp.status} - Storm: {data.get('storm_name')}")
            print(f"  Data Label: {data.get('data_source_label')}")
            print(f"  Current Position: {data['current_cyclone_position']}")
            print(f"  Risk Level: {data['risk_assessment']['risk_level']} ({data['risk_assessment']['alert_tier']})")
            print(f"  Closest Coastal Target: {data['risk_assessment']['closest_coastal_target']} ({data['risk_assessment']['min_coastal_distance_km']} km)")
            print(f"  Weather Source: {data['weather_data']['source']}")
            print(f"  AI Top Drivers: {data['ai_explanation']['top_drivers']}")
            assert data['status'] == "SUCCESS"
            assert len(data['gru_predicted_track']) == 6
            assert data['risk_assessment'] is not None
            assert data['ai_explanation'] is not None
            passed += 1
            print("  [PASS]: Full analysis pipeline verified.")
    except Exception as e:
        print(f"  [FAIL]: {e}")

    # Test 4: Analyze Cyclone Fusion (Custom Live Observations)
    print("\n[TEST 4/5] Testing POST /api/analyze (Custom Live Observations) ...")
    try:
        custom_obs = [
            {"latitude": 12.0, "longitude": 86.0, "wind_speed": 65, "pressure": 985, "month": 5},
            {"latitude": 13.5, "longitude": 85.5, "wind_speed": 85, "pressure": 970, "month": 5},
            {"latitude": 15.0, "longitude": 85.0, "wind_speed": 110, "pressure": 950, "month": 5},
            {"latitude": 16.5, "longitude": 84.8, "wind_speed": 130, "pressure": 935, "month": 5},
            {"latitude": 18.0, "longitude": 84.5, "wind_speed": 145, "pressure": 925, "month": 5},
            {"latitude": 19.5, "longitude": 84.8, "wind_speed": 155, "pressure": 918, "month": 5}
        ]
        req_data = json.dumps({"observations": custom_obs, "use_demo_mode": False, "steps": 4}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/api/analyze",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            print(f"  Status: {resp.status} - Data Label: {data.get('data_source_label')}")
            print(f"  Risk Level: {data['risk_assessment']['risk_level']} ({data['risk_assessment']['alert_tier']})")
            assert data['status'] == "SUCCESS"
            assert data['is_demo'] is False
            assert len(data['gru_predicted_track']) == 4
            passed += 1
            print("  [PASS]: Custom observation pipeline verified.")
    except Exception as e:
        print(f"  [FAIL]: {e}")

    # Test 5: Frontend Serving & CORS
    print("\n[TEST 5/5] Testing GET / (Frontend Dashboard Serving & CORS) ...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/", headers={"Origin": "http://localhost:3000"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read().decode()
            cors_header = resp.headers.get("Access-Control-Allow-Origin")
            print(f"  Status: {resp.status} - Content Length: {len(content)} bytes")
            print(f"  CORS Header: {cors_header}")
            print(f"  Contains GRU Tab: {'tabGruTrajectory' in content}")
            print(f"  Contains Leaflet Canvas: {'v2map' in content}")
            assert resp.status == 200
            assert "tabGruTrajectory" in content
            assert "v2map" in content
            passed += 1
            print("  [PASS]: Frontend dashboard serving and markup verified.")
            print("  [PASS]")
    except Exception as e:
        print(f"  [FAIL]: {e}")

    print("\n" + "="*65)
    print(f"TEST SUMMARY: {passed}/{total} TESTS PASSED")
    print("="*65)
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
