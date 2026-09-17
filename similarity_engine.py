class CycloneSimilarityEngine:
    def __init__(self):
        self.historical_database = [
            {
                "name": "Super Cyclone (1999)",
                "year": 1999,
                "basin": "Bay of Bengal",
                "peak_msw_knots": 140,
                "min_pressure_hpa": 912,
                "landfall_district": "Jagatsinghpur, Odisha"
            },
            {
                "name": "Extremely Severe Cyclone Phailin",
                "year": 2013,
                "basin": "Bay of Bengal",
                "peak_msw_knots": 115,
                "min_pressure_hpa": 940,
                "landfall_district": "Gopalpur, Odisha"
            },
            {
                "name": "Extremely Severe Cyclone Fani",
                "year": 2019,
                "basin": "Bay of Bengal",
                "peak_msw_knots": 115,
                "min_pressure_hpa": 937,
                "landfall_district": "Puri, Odisha"
            },
            {
                "name": "Super Cyclone Amphan",
                "year": 2020,
                "basin": "Bay of Bengal",
                "peak_msw_knots": 130,
                "min_pressure_hpa": 920,
                "landfall_district": "Sundarbans, West Bengal"
            },
            {
                "name": "Extremely Severe Cyclone Biparjoy",
                "year": 2023,
                "basin": "Arabian Sea",
                "peak_msw_knots": 90,
                "min_pressure_hpa": 958,
                "landfall_district": "Jakhau Port, Gujarat"
            }
        ]

    def find_top_analogs(self, current_wind, current_pressure, current_basin="Bay of Bengal"):
        results = []
        for hist in self.historical_database:
            basin_match = 1.0 if hist["basin"] == current_basin else 0.6
            wind_diff = abs(hist["peak_msw_knots"] - current_wind)
            press_diff = abs(hist["min_pressure_hpa"] - current_pressure)
            
            wind_score = max(0.0, 1.0 - (wind_diff / 80.0))
            press_score = max(0.0, 1.0 - (press_diff / 60.0))
            similarity = (wind_score * 0.5 + press_score * 0.3 + basin_match * 0.2) * 100.0
            
            results.append({
                "cyclone_name": hist["name"],
                "year": hist["year"],
                "similarity_score_pct": round(similarity, 1),
                "historical_peak_wind_knots": hist["peak_msw_knots"],
                "landfall_location": hist["landfall_district"]
            })

        results.sort(key=lambda x: x["similarity_score_pct"], reverse=True)
        return results[:3]
