"""
CycloVision AI - Trajectory Fusion & Risk Assessment Service
Couples:
1. Historical / Real-Time Observation Sequences
2. Trained GRU Trajectory Model (models/trajectory/trajectory_model.keras)
3. Live Weather Integration (Open-Meteo)
4. Transparent Geodesic Risk Assessment
5. Empirical Gradient Feature Sensitivity (XAI)
6. Ground-Truth DEMO MODE with FANI historical data
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.trajectory.predict import TrajectoryPredictor, haversine_distance_km, clamp_coordinates
from v2.services.weather_service import get_weather_provider

DEMO_DATA_PATH = REPO_ROOT / "reports" / "trajectory" / "test_prediction.json"

# Key coastal monitoring centers for risk evaluation
COASTAL_TARGETS = [
    {"name": "Puri", "state": "Odisha", "lat": 19.81, "lon": 85.83, "population": "200k+"},
    {"name": "Bhubaneswar", "state": "Odisha", "lat": 20.29, "lon": 85.82, "population": "1M+"},
    {"name": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.68, "lon": 83.21, "population": "2M+"},
    {"name": "Paradip Port", "state": "Odisha", "lat": 20.26, "lon": 86.66, "population": "Major Port"},
    {"name": "Kolkata / Sundarbans", "state": "West Bengal", "lat": 22.57, "lon": 88.36, "population": "14M+"},
    {"name": "Chennai", "state": "Tamil Nadu", "lat": 13.08, "lon": 80.27, "population": "10M+"}
]


class TrajectoryFusionService:
    def __init__(self):
        self.predictor = None
        self._load_predictor()

    def _load_predictor(self):
        try:
            self.predictor = TrajectoryPredictor()
        except Exception as e:
            print(f"[WARN] Could not initialize TrajectoryPredictor: {e}")
            self.predictor = None

    def get_demo_fani_data(self) -> Dict[str, Any]:
        """Loads verified ground-truth historical FANI trajectory test data."""
        if DEMO_DATA_PATH.exists():
            with open(DEMO_DATA_PATH, "r") as f:
                return json.load(f)
        # Fallback in-memory FANI track if file is missing
        return {
            "storm_id": "2019116N02090",
            "storm_name": "FANI",
            "past_track_input_6_steps": [
                {"iso_time": "2019-04-27 00:00:00", "latitude": 4.6, "longitude": 89.4, "wind_speed": 30.0, "pressure": 997.0, "month": 4},
                {"iso_time": "2019-04-27 03:00:00", "latitude": 4.9, "longitude": 89.4, "wind_speed": 30.0, "pressure": 996.0, "month": 4},
                {"iso_time": "2019-04-27 06:00:00", "latitude": 5.1, "longitude": 89.4, "wind_speed": 35.0, "pressure": 995.0, "month": 4},
                {"iso_time": "2019-04-27 09:00:00", "latitude": 5.5, "longitude": 89.3, "wind_speed": 40.0, "pressure": 994.0, "month": 4},
                {"iso_time": "2019-04-27 12:00:00", "latitude": 6.0, "longitude": 89.2, "wind_speed": 45.0, "pressure": 992.0, "month": 4},
                {"iso_time": "2019-04-27 15:00:00", "latitude": 6.4, "longitude": 88.9, "wind_speed": 45.0, "pressure": 992.0, "month": 4}
            ]
        }

    def compute_feature_sensitivity(self, observations: list) -> Dict[str, float]:
        """
        Computes empirical input feature sensitivity (finite-difference gradient)
        w.r.t the trained GRU model. Never invents values.
        """
        if not self.predictor:
            return {}

        try:
            feat_names = self.predictor.features
            raw_rows = []
            for obs in observations:
                lat = float(obs.get('latitude', obs.get('LAT', 15.0)))
                lon = float(obs.get('longitude', obs.get('LON', 85.0)))
                wind = float(obs.get('wind_speed', obs.get('WIND', 45.0)))
                pres = float(obs.get('pressure', obs.get('PRES', 995.0)))
                month = int(obs.get('month', 5))
                raw_rows.append({'LAT': lat, 'LON': lon, 'WIND': wind, 'PRES': pres, 'month': month})

            while len(raw_rows) < self.predictor.input_seq_len:
                raw_rows.insert(0, raw_rows[0].copy())
            recent_rows = raw_rows[-self.predictor.input_seq_len:]

            feat_matrix = []
            for i in range(len(recent_rows)):
                curr = recent_rows[i]
                prev = recent_rows[i - 1] if i > 0 else recent_rows[0]
                row_feats = [
                    curr['LAT'],
                    curr['LON'],
                    curr['LAT'] - prev['LAT'],
                    curr['LON'] - prev['LON'],
                    curr['WIND'],
                    curr['PRES'],
                    np.sin(2 * np.pi * curr['month'] / 12.0),
                    np.cos(2 * np.pi * curr['month'] / 12.0)
                ]
                feat_matrix.append(row_feats)

            X = np.array([feat_matrix], dtype=np.float32)
            X_scaled = self.predictor.feature_scaler.transform(X.reshape(-1, len(feat_names))).reshape(1, self.predictor.input_seq_len, len(feat_names))
            base_pred = self.predictor.model.predict(X_scaled, verbose=0)

            eps = 0.1
            sensitivities = {}
            for f_idx, f_name in enumerate(feat_names):
                X_pert = X_scaled.copy()
                X_pert[:, :, f_idx] += eps
                pert_pred = self.predictor.model.predict(X_pert, verbose=0)
                diff = float(np.linalg.norm(pert_pred - base_pred))
                sensitivities[f_name] = diff

            total_sens = sum(sensitivities.values()) or 1.0
            return {k: round(v / total_sens * 100, 1) for k, v in sensitivities.items()}
        except Exception as e:
            print(f"[WARN] Sensitivity computation error: {e}")
            return {}

    def assess_coastal_risk(
        self,
        current_lat: float,
        current_lon: float,
        wind_kts: float,
        pressure_hpa: float,
        predicted_points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transparent prototype risk assessment based only on real available physics:
        - Haversine distance from track to coast/cities
        - Wind speed tier (IMD intensity scale)
        - Pressure anomaly (1013 - pressure)
        """
        wind_kmph = round(wind_kts * 1.852, 1)
        pressure_drop = max(0.0, 1013.25 - pressure_hpa)

        # Track trajectory points (current + predicted)
        all_track = [(current_lat, current_lon)] + [(p['latitude'], p['longitude']) for p in predicted_points]

        # Evaluate distance to all coastal targets
        target_assessments = []
        min_distance_overall = 99999.0
        closest_city = None

        for city in COASTAL_TARGETS:
            c_lat, c_lon = city['lat'], city['lon']
            dists = [haversine_distance_km(t_lat, t_lon, c_lat, c_lon) for t_lat, t_lon in all_track]
            min_d = min(dists)
            if min_d < min_distance_overall:
                min_distance_overall = min_d
                closest_city = city['name']

            target_assessments.append({
                "city": city['name'],
                "state": city['state'],
                "min_track_distance_km": round(min_d, 1),
                "direct_threat": bool(min_d <= 150.0)
            })

        # Sort by distance
        target_assessments.sort(key=lambda x: x['min_track_distance_km'])

        # Transparent Risk Categorization
        if min_distance_overall <= 100.0 and wind_kmph >= 165.0:
            level = "EXTREME"
            alert_tier = "RED ALERT"
            action = "Mandatory mass evacuation within 100km coastal buffer. Complete port suspension."
        elif min_distance_overall <= 200.0 and wind_kmph >= 115.0:
            level = "HIGH"
            alert_tier = "ORANGE ALERT"
            action = "Shelter activation and pre-positioning of NDRF/SDRF teams. Fishermen strictly prohibited at sea."
        elif min_distance_overall <= 350.0:
            level = "MODERATE"
            alert_tier = "YELLOW ALERT"
            action = "Heightened state surveillance. Ports put on local cautionary signal LC-3."
        else:
            level = "LOW"
            alert_tier = "GREEN / WATCH"
            action = "Routine monitoring. Deep sea vessels advised to exercise caution."

        factors = [
            f"Closest coastal approach: {round(min_distance_overall, 1)} km from {closest_city}",
            f"Maximum sustained wind velocity: {wind_kmph} km/h ({round(wind_kts, 1)} knots)",
            f"Central barometric pressure drop: {round(pressure_drop, 1)} hPa below standard atmosphere"
        ]

        return {
            "risk_level": level,
            "alert_tier": alert_tier,
            "recommended_action": action,
            "closest_coastal_target": closest_city,
            "min_coastal_distance_km": round(min_distance_overall, 1),
            "wind_kmph": wind_kmph,
            "pressure_hpa": pressure_hpa,
            "factors": factors,
            "target_impact_matrix": target_assessments[:4]
        }

    def analyze_cyclone(
        self,
        storm_id: Optional[str] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        use_demo_mode: bool = False,
        steps: int = 6,
        include_weather: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-task cyclone fusion pipeline:
        Observations -> Data Fusion -> GRU Prediction -> Risk Assessment -> Weather -> XAI.
        """
        if not self.predictor:
            self._load_predictor()
            if not self.predictor:
                raise RuntimeError("GRU Trajectory model could not be loaded. Verify models/trajectory/ files.")

        # Determine Demo Mode vs Live Observations
        is_demo = False
        data_source_label = "OPERATOR / LIVE OBSERVATIONS"
        storm_name = "ACTIVE CYCLONE TARGET"

        if use_demo_mode or not observations or len(observations) == 0:
            from v2.data.ibtracs_loader import get_storm_by_id
            storm_record = get_storm_by_id(storm_id) if storm_id else None

            if storm_record and storm_id in ["AMPHAN_2020", "BIPARJOY_2023"]:
                storm_name = f"{storm_record['name']} ({storm_record['season']})"
                observations = []
                for ob in storm_record.get("observations", []):
                    m = 5
                    if "time" in ob:
                        try:
                            m = int(ob["time"].split("-")[1])
                        except Exception:
                            m = 5
                    observations.append({
                        "iso_time": ob.get("time", ""),
                        "latitude": float(ob.get("lat", 0.0)),
                        "longitude": float(ob.get("lon", 0.0)),
                        "wind_speed": float(ob.get("wind_kts", 40.0)),
                        "pressure": float(ob.get("pres", 995.0)),
                        "month": m
                    })
                is_demo = True
                data_source_label = f"DEMO / HISTORICAL DATA (NOAA IBTrACS Cyclone {storm_record['name']})"
            else:
                # Fallback to verified ground-truth Cyclone FANI historical dataset
                demo_obj = self.get_demo_fani_data()
                observations = demo_obj.get("past_track_input_6_steps", [])
                storm_name = demo_obj.get("storm_name", "FANI (2019)")
                is_demo = True
                data_source_label = "DEMO / HISTORICAL DATA (NOAA IBTrACS Cyclone Fani)"

        if len(observations) == 0:
            raise ValueError("No cyclone observations available to analyze.")

        # Current state is latest observation
        latest_obs = observations[-1]
        cur_lat = float(latest_obs.get('latitude', latest_obs.get('LAT', 15.0)))
        cur_lon = float(latest_obs.get('longitude', latest_obs.get('LON', 85.0)))
        cur_wind = float(latest_obs.get('wind_speed', latest_obs.get('WIND', latest_obs.get('wind_kts', 45.0))))
        cur_pres = float(latest_obs.get('pressure', latest_obs.get('PRES', latest_obs.get('pressure_hpa', 995.0))))

        # 1. GRU Trajectory Prediction
        raw_preds = self.predictor.predict(observations, steps=steps)
        predicted_track = []
        for p in raw_preds:
            s = p['step']
            lead_h = s * 6  # 6h per step
            # Progressive empirical uncertainty cone radius based on measured test error (21km at step 1 to 96km at step 6)
            err_cone_km = round(21.5 + (s - 1) * 15.0, 1)
            predicted_track.append({
                "step": s,
                "lead_hours": f"+{lead_h}h",
                "latitude": p['latitude'],
                "longitude": p['longitude'],
                "uncertainty_radius_km": err_cone_km
            })

        # 2. Weather Data Integration (Open-Meteo)
        weather_info = None
        if include_weather:
            try:
                weather_prov = get_weather_provider()
                weather_info = weather_prov.get_current_weather(cur_lat, cur_lon)
            except Exception as w_err:
                weather_info = {
                    "source": "Open-Meteo (Offline / Fallback)",
                    "status": f"Unavailable: {w_err}",
                    "temperature_c": None,
                    "wind_speed_kmh": None,
                    "surface_pressure_hpa": None
                }

        # 3. Transparent Coastal Risk Assessment
        risk_res = self.assess_coastal_risk(
            current_lat=cur_lat,
            current_lon=cur_lon,
            wind_kts=cur_wind,
            pressure_hpa=cur_pres,
            predicted_points=predicted_track
        )

        # 4. Empirical Feature Sensitivity (XAI)
        feat_importance = self.compute_feature_sensitivity(observations)

        # Build Explanation
        ai_explanation = {
            "methodology": "Empirical Finite-Difference Sensitivity (dTrajectory / dFeature)",
            "description": "Calculates the magnitude of future coordinate displacement when each input physical feature is perturbed. Reflects genuine GRU network gradient response without synthetic interpolation.",
            "feature_importance_pct": feat_importance,
            "top_drivers": sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:4],
            "synoptic_rationale": (
                f"The trajectory forecast is primarily governed by recent geographic coordinates and "
                f"motion velocity vectors (DELTA_LON: {feat_importance.get('DELTA_LON', 0)}%, "
                f"DELTA_LAT: {feat_importance.get('DELTA_LAT', 0)}%). Central pressure ({cur_pres} hPa) "
                f"and wind intensity ({cur_wind} kts) modulate steering response."
            )
        }

        # Format historical track points
        historical_track = []
        for idx, o in enumerate(observations):
            historical_track.append({
                "step": idx + 1,
                "latitude": float(o.get('latitude', o.get('LAT', 0.0))),
                "longitude": float(o.get('longitude', o.get('LON', 0.0))),
                "wind_kts": float(o.get('wind_speed', o.get('WIND', 0.0))),
                "pressure_hpa": float(o.get('pressure', o.get('PRES', 1000.0))),
                "iso_time": o.get('iso_time', f"T-{len(observations) - idx}h")
            })

        return {
            "status": "SUCCESS",
            "data_source_label": data_source_label,
            "is_demo": is_demo,
            "storm_name": storm_name,
            "current_cyclone_position": {
                "latitude": cur_lat,
                "longitude": cur_lon,
                "wind_kts": cur_wind,
                "wind_kmph": round(cur_wind * 1.852, 1),
                "pressure_hpa": cur_pres,
                "stage": "Very Severe Cyclonic Storm" if cur_wind >= 64 else "Cyclonic Storm"
            },
            "historical_track": historical_track,
            "gru_predicted_track": predicted_track,
            "prediction_points_count": len(predicted_track),
            "weather_data": weather_info,
            "risk_assessment": risk_res,
            "ai_explanation": ai_explanation
        }


# Singleton service
_fusion_service = None

def get_trajectory_fusion_service() -> TrajectoryFusionService:
    global _fusion_service
    if _fusion_service is None:
        _fusion_service = TrajectoryFusionService()
    return _fusion_service
