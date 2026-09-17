COASTAL_SECTORS = [
    {"sector": "North Odisha", "districts": ["Balasore", "Bhadrak", "Kendrapara"], "lat_range": [20.5, 21.8], "lon_range": [86.5, 87.5]},
    {"sector": "Central Odisha", "districts": ["Puri", "Jagatsinghpur", "Ganjam"], "lat_range": [19.2, 20.4], "lon_range": [84.8, 86.4]},
    {"sector": "North Andhra", "districts": ["Srikakulam", "Vizianagaram", "Visakhapatnam"], "lat_range": [17.5, 19.1], "lon_range": [83.0, 84.7]},
    {"sector": "South Andhra", "districts": ["East Godavari", "West Godavari", "Krishna"], "lat_range": [15.5, 17.4], "lon_range": [80.5, 82.9]}
]

def compute_gis_landfall(track_forecast, current_coords=None):
    if current_coords:
        cur_lat = current_coords.get("lat")
        cur_lon = current_coords.get("lon")
        if cur_lat is not None and cur_lon is not None:
            for sec in COASTAL_SECTORS:
                if sec["lat_range"][0] <= cur_lat <= sec["lat_range"][1] and sec["lon_range"][0] <= cur_lon <= sec["lon_range"][1]:
                    return {
                        "landfall_detected": True,
                        "estimated_lead_time_hours": 0,
                        "landfall_coordinates": {"lat": cur_lat, "lon": cur_lon},
                        "coastal_sector": sec["sector"],
                        "vulnerable_districts": sec["districts"],
                        "expected_storm_surge_m": 3.8,
                        "status": "IMMINENT_OR_ONGOING_LANDFALL"
                    }

    for pt in track_forecast:
        lat, lon = pt["latitude"], pt["longitude"]
        for sec in COASTAL_SECTORS:
            if sec["lat_range"][0] <= lat <= sec["lat_range"][1] and sec["lon_range"][0] <= lon <= sec["lon_range"][1]:
                return {
                    "landfall_detected": True,
                    "estimated_lead_time_hours": pt["lead_hour"],
                    "landfall_coordinates": {"lat": lat, "lon": lon},
                    "coastal_sector": sec["sector"],
                    "vulnerable_districts": sec["districts"],
                    "expected_storm_surge_m": 3.8,
                    "status": "CRITICAL_COASTAL_INTERSECTION"
                }
    return {"landfall_detected": False, "status": "REMAINING_OVER_OPEN_OCEAN"}

