# NOAA IBTrACS v4 + IMD RSMC Best-Track Loader
HISTORICAL_STORMS_DATABASE = {
    "FANI_2019": {
        "storm_id": "2019116N11086",
        "name": "FANI",
        "season": 2019,
        "basin": "Bay of Bengal",
        "category": "Extremely Severe Cyclonic Storm (ESCS)",
        "peak_wind_kts": 115,
        "peak_wind_kmph": 215,
        "min_pressure_hpa": 932,
        "landfall_date": "2019-05-03 08:00 UTC",
        "landfall_location": "Puri, Odisha (19.8N, 85.8E)",
        "observations": [
            {"time": "2019-04-27 06:00", "lat": 5.2, "lon": 88.5, "wind_kts": 30, "pres": 1000, "stage": "Depression"},
            {"time": "2019-04-29 12:00", "lat": 8.6, "lon": 86.9, "wind_kts": 55, "pres": 990, "stage": "Cyclonic Storm"},
            {"time": "2019-05-01 18:00", "lat": 14.5, "lon": 84.1, "wind_kts": 95, "pres": 954, "stage": "VSCS"},
            {"time": "2019-05-02 12:00", "lat": 17.1, "lon": 84.8, "wind_kts": 115, "pres": 932, "stage": "ESCS"},
            {"time": "2019-05-03 08:00", "lat": 19.8, "lon": 85.8, "wind_kts": 100, "pres": 945, "stage": "Landfall (Puri)"}
        ]
    },
    "AMPHAN_2020": {
        "storm_id": "2020136N10087",
        "name": "AMPHAN",
        "season": 2020,
        "basin": "Bay of Bengal",
        "category": "Super Cyclonic Storm (SuCS)",
        "peak_wind_kts": 140,
        "peak_wind_kmph": 260,
        "min_pressure_hpa": 907,
        "landfall_date": "2020-05-20 11:30 UTC",
        "landfall_location": "Bakkhali, West Bengal (21.7N, 88.3E)",
        "observations": [
            {"time": "2020-05-16 00:00", "lat": 10.4, "lon": 87.0, "wind_kts": 35, "pres": 998, "stage": "Deep Depression"},
            {"time": "2020-05-17 12:00", "lat": 12.0, "lon": 86.0, "wind_kts": 65, "pres": 980, "stage": "Cyclonic Storm"},
            {"time": "2020-05-18 06:00", "lat": 13.4, "lon": 86.2, "wind_kts": 140, "pres": 907, "stage": "Super Cyclone (SuCS)"},
            {"time": "2020-05-20 11:30", "lat": 21.7, "lon": 88.3, "wind_kts": 85, "pres": 958, "stage": "Landfall (WB)"}
        ]
    },
    "BIPARJOY_2023": {
        "storm_id": "2023157N12066",
        "name": "BIPARJOY",
        "season": 2023,
        "basin": "Arabian Sea",
        "category": "Extremely Severe Cyclonic Storm (ESCS)",
        "peak_wind_kts": 90,
        "peak_wind_kmph": 165,
        "min_pressure_hpa": 958,
        "landfall_date": "2023-06-15 17:30 UTC",
        "landfall_location": "Jakhau Port, Gujarat (23.2N, 68.6E)",
        "observations": [
            {"time": "2023-06-06 06:00", "lat": 12.1, "lon": 66.0, "wind_kts": 35, "pres": 998, "stage": "Deep Depression"},
            {"time": "2023-06-08 12:00", "lat": 14.2, "lon": 66.2, "wind_kts": 70, "pres": 976, "stage": "VSCS"},
            {"time": "2023-06-11 18:00", "lat": 18.6, "lon": 67.7, "wind_kts": 90, "pres": 958, "stage": "ESCS"},
            {"time": "2023-06-15 17:30", "lat": 23.2, "lon": 68.6, "wind_kts": 65, "pres": 978, "stage": "Landfall (Jakhau)"}
        ]
    }
}

def get_storm_by_id(storm_id):
    return HISTORICAL_STORMS_DATABASE.get(storm_id)

def list_available_storms():
    return [
        {"id": k, "name": v["name"], "season": v["season"], "basin": v["basin"], "peak_wind": f"{v['peak_wind_kmph']} km/h"}
        for k, v in HISTORICAL_STORMS_DATABASE.items()
    ]
