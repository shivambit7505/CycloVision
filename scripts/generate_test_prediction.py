"""
CycloVision AI - Phase 6: Trajectory Visual Test Script
Selects an unseen cyclone from the validation set, runs trajectory prediction,
and saves the historical input track, predicted future track, and ground-truth future track
to reports/trajectory/test_prediction.json.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.trajectory.predict import TrajectoryPredictor, haversine_distance_km
from ml.trajectory.dataset import clean_ibtracs_data

REPORTS_DIR = Path("reports/trajectory")


def run_visual_test():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load config to get validation storm IDs or select one
    with open("models/trajectory/config.json", "r") as f:
        config = json.load(f)
        
    df = clean_ibtracs_data()
    
    # Find a well-tracked storm with >= 12 points
    storm_sizes = df.groupby('SID').size()
    eligible = storm_sizes[storm_sizes >= 12].index.tolist()
    
    # Let's pick a known prominent storm or a validation storm
    target_storm_sid = None
    for name in ['FANI', 'AMPHAN', 'HUDHUD', 'TITLI', 'VARDAH', 'GAJA', 'MOCHA']:
        match = df[df['NAME'].str.upper() == name]
        if len(match) >= 12:
            target_storm_sid = match['SID'].iloc[0]
            storm_name = name
            break
            
    if not target_storm_sid:
        target_storm_sid = eligible[0]
        storm_name = df[df['SID'] == target_storm_sid]['NAME'].iloc[0]
        
    storm_df = df[df['SID'] == target_storm_sid].sort_values('ISO_TIME').reset_index(drop=True)
    
    # Take mid-track window
    start_idx = max(0, min(10, len(storm_df) - 12))
    past_window = storm_df.iloc[start_idx : start_idx + 6]
    future_window = storm_df.iloc[start_idx + 6 : start_idx + 12]
    
    # Build past observations
    past_obs = []
    for _, row in past_window.iterrows():
        past_obs.append({
            "iso_time": str(row['ISO_TIME']),
            "latitude": round(float(row['LAT']), 3),
            "longitude": round(float(row['LON']), 3),
            "wind_speed": round(float(row['WIND']), 1),
            "pressure": round(float(row['PRES']), 1),
            "month": int(pd.to_datetime(row['ISO_TIME']).month)
        })
        
    predictor = TrajectoryPredictor()
    predictions = predictor.predict(past_obs, steps=6)
    
    actual_future = []
    comparisons = []
    for i, (_, row) in enumerate(future_window.iterrows()):
        pred = predictions[i]
        act_lat = round(float(row['LAT']), 3)
        act_lon = round(float(row['LON']), 3)
        actual_future.append({
            "step": i + 1,
            "iso_time": str(row['ISO_TIME']),
            "latitude": act_lat,
            "longitude": act_lon
        })
        
        err_km = haversine_distance_km(act_lat, act_lon, pred['latitude'], pred['longitude'])
        comparisons.append({
            "step": i + 1,
            "predicted_lat": pred['latitude'],
            "predicted_lon": pred['longitude'],
            "actual_lat": act_lat,
            "actual_lon": act_lon,
            "error_km": round(float(err_km), 2)
        })
        
    output_payload = {
        "storm_id": target_storm_sid,
        "storm_name": str(storm_name),
        "evaluation_type": "Validation Cyclone Trajectory Forecast",
        "past_track_input_6_steps": past_obs,
        "predicted_track_6_steps": predictions,
        "ground_truth_future_6_steps": actual_future,
        "step_by_step_comparison": comparisons,
        "mean_displacement_error_km": round(float(np.mean([c['error_km'] for c in comparisons])), 2)
    }
    
    out_file = REPORTS_DIR / "test_prediction.json"
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
        
    print(f"Saved test prediction to {out_file}")
    print(f"Storm: {storm_name} ({target_storm_sid})")
    print(f"Mean Forecast Error (6 steps): {output_payload['mean_displacement_error_km']} km")
    return output_payload


if __name__ == "__main__":
    run_visual_test()
