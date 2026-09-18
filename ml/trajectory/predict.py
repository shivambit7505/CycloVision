"""
CycloVision AI - Trajectory Prediction Inference Engine
Accepts recent historical cyclone observations (up to 6 past steps)
and predicts the next 3 to 6 future geographic positions (latitude, longitude).
Ensures predicted coordinates are clamped to valid physical geographical bounds.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["KERAS_BACKEND"] = "torch"
import keras

MODEL_PATH = Path("models/trajectory/trajectory_model.keras")
SCALER_PATH = Path("models/trajectory/scaler.pkl")


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between two coordinate sets in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    return R * c


def clamp_coordinates(lat: float, lon: float):
    """Clamps coordinates to valid Earth latitude and longitude boundaries."""
    c_lat = max(-90.0, min(90.0, float(lat)))
    # Wrap or clamp lon
    if lon > 180.0:
        lon = lon - 360.0
    elif lon < -180.0:
        lon = lon + 360.0
    c_lon = max(-180.0, min(180.0, float(lon)))
    return round(c_lat, 4), round(c_lon, 4)


class TrajectoryPredictor:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        m_p = Path(model_path) if model_path else MODEL_PATH
        s_p = Path(scaler_path) if scaler_path else SCALER_PATH
        
        if not m_p.exists():
            raise FileNotFoundError(f"Model file not found at {m_p}. Train model first.")
        if not s_p.exists():
            raise FileNotFoundError(f"Scaler file not found at {s_p}. Train model first.")
            
        self.model = keras.models.load_model(str(m_p))
        with open(s_p, "rb") as f:
            self.scalers = pickle.load(f)
            
        self.feature_scaler = self.scalers["feature_scaler"]
        self.target_scaler = self.scalers["target_scaler"]
        self.features = self.scalers["features"]
        self.input_seq_len = self.scalers["input_seq_len"]
        self.output_seq_len = self.scalers["output_seq_len"]

    def predict(self, observations: list, steps: int = 6) -> list:
        """
        Predicts next N steps given a list of past observation dicts.
        Each observation should contain at minimum:
          - 'latitude' (or 'LAT')
          - 'longitude' (or 'LON')
          - optionally 'wind_speed' (or 'WIND')
          - optionally 'pressure' (or 'PRES')
          - optionally 'month' (1-12)
          
        Args:
            observations: list of at least 1 and up to 6 observation dicts.
            steps: number of future positions to return (between 1 and 6, default 6).
            
        Returns:
            List of dicts: [{"step": 1, "latitude": 18.2, "longitude": 85.4}, ...]
        """
        if not observations or len(observations) == 0:
            raise ValueError("Observations list cannot be empty.")
            
        steps = max(1, min(self.output_seq_len, int(steps)))
        
        # Pad or slice to exactly input_seq_len (6 observations)
        # If fewer than 6, repeat the first observation backward to preserve motion continuity
        raw_rows = []
        for obs in observations:
            lat = float(obs.get('latitude', obs.get('LAT', 15.0)))
            lon = float(obs.get('longitude', obs.get('LON', 85.0)))
            wind = float(obs.get('wind_speed', obs.get('WIND', obs.get('wind', 45.0))))
            pres = float(obs.get('pressure', obs.get('PRES', obs.get('pres', 995.0))))
            month = int(obs.get('month', 5))
            raw_rows.append({
                'LAT': lat,
                'LON': lon,
                'WIND': wind,
                'PRES': pres,
                'month': month
            })
            
        while len(raw_rows) < self.input_seq_len:
            raw_rows.insert(0, raw_rows[0].copy())
            
        # Take the most recent 6 observations
        recent_rows = raw_rows[-self.input_seq_len:]
        
        # Build feature matrix
        feat_matrix = []
        for i in range(len(recent_rows)):
            curr = recent_rows[i]
            prev = recent_rows[i - 1] if i > 0 else recent_rows[0]
            
            d_lat = curr['LAT'] - prev['LAT']
            d_lon = curr['LON'] - prev['LON']
            m = curr['month']
            m_sin = np.sin(2 * np.pi * m / 12.0)
            m_cos = np.cos(2 * np.pi * m / 12.0)
            
            row_feats = [
                curr['LAT'],
                curr['LON'],
                d_lat,
                d_lon,
                curr['WIND'],
                curr['PRES'],
                m_sin,
                m_cos
            ]
            feat_matrix.append(row_feats)
            
        # Scale input
        X = np.array([feat_matrix], dtype=np.float32)  # (1, 6, 8)
        X_scaled = self.feature_scaler.transform(X.reshape(-1, len(self.features))).reshape(1, self.input_seq_len, len(self.features))
        
        # Model forward pass
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)  # (1, 6, 2)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 2)).reshape(self.output_seq_len, 2)
        
        results = []
        for s in range(steps):
            raw_lat, raw_lon = y_pred[s, 0], y_pred[s, 1]
            valid_lat, valid_lon = clamp_coordinates(raw_lat, raw_lon)
            results.append({
                "step": s + 1,
                "latitude": valid_lat,
                "longitude": valid_lon
            })
            
        return results


def main():
    parser = argparse.ArgumentParser(description="CycloVision AI - Cyclone Trajectory Predictor")
    parser.add_argument("--input_json", type=str, help="Path to JSON file containing list of past observations")
    parser.add_argument("--steps", type=int, default=6, help="Number of steps to forecast (default: 6)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Model file path")
    parser.add_argument("--scaler", type=str, default=str(SCALER_PATH), help="Scaler file path")
    args = parser.parse_args()
    
    predictor = TrajectoryPredictor(args.model, args.scaler)
    
    if args.input_json and Path(args.input_json).exists():
        with open(args.input_json, "r") as f:
            obs = json.load(f)
    else:
        # Default sample observation sequence (e.g. Cyclone Fani track sequence in Bay of Bengal)
        print("No input JSON specified; using sample Bay of Bengal cyclone track observations...")
        obs = [
            {"step": 1, "latitude": 9.8, "longitude": 87.2, "wind_speed": 45, "pressure": 998, "month": 5},
            {"step": 2, "latitude": 10.5, "longitude": 86.8, "wind_speed": 55, "pressure": 990, "month": 5},
            {"step": 3, "latitude": 11.4, "longitude": 86.2, "wind_speed": 70, "pressure": 978, "month": 5},
            {"step": 4, "latitude": 12.6, "longitude": 85.7, "wind_speed": 90, "pressure": 960, "month": 5},
            {"step": 5, "latitude": 14.1, "longitude": 85.2, "wind_speed": 115, "pressure": 940, "month": 5},
            {"step": 6, "latitude": 16.0, "longitude": 85.0, "wind_speed": 130, "pressure": 932, "month": 5}
        ]
        
    preds = predictor.predict(obs, steps=args.steps)
    print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
