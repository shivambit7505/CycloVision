"""
CycloVision AI - Trajectory Prediction Validation Script
Evaluates trained GRU model strictly on unseen validation cyclone tracks.
Computes real-world geographic metrics:
- Latitude MAE (degrees)
- Longitude MAE (degrees)
- Great-Circle Haversine Position Error (km) per forecast horizon step and overall
- Validation Loss (scaled MSE)
"""

import os
import json
import pickle
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["KERAS_BACKEND"] = "torch"
import keras

from ml.trajectory.dataset import generate_sequences

MODEL_PATH = Path("models/trajectory/trajectory_model.keras")
SCALER_PATH = Path("models/trajectory/scaler.pkl")
CONFIG_PATH = Path("models/trajectory/config.json")
REPORTS_DIR = Path("reports/trajectory")


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between two coordinate sets in kilometers.
    """
    R = 6371.0  # Earth's mean radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    return R * c


def validate_trajectory_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[1/3] Loading trained trajectory model and preprocessing scalers...")
    model = keras.models.load_model(str(MODEL_PATH))
    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)
        
    print("[2/3] Generating unseen validation split by Cyclone SID...")
    (
        _, _,
        X_val_scaled, y_val_scaled,
        _, y_val_raw,
        _,
        train_sids, val_sids
    ) = generate_sequences(val_split=0.20, random_seed=42)
    
    print(f"  Validation sequences: {len(X_val_scaled)} across {len(val_sids)} unseen cyclone tracks")
    
    # Compute scaled validation loss
    eval_res = model.evaluate(X_val_scaled, y_val_scaled, verbose=0)
    val_loss = float(eval_res[0] if isinstance(eval_res, list) else eval_res)
    
    # Run predictions
    print("[3/3] Predicting trajectories and calculating physical coordinate errors...")
    y_pred_scaled = model.predict(X_val_scaled, batch_size=128, verbose=0)
    
    # Invert scaling to real geographic coordinates (°N, °E)
    target_scaler = scalers["target_scaler"]
    N_val, S_out, _ = y_pred_scaled.shape
    
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 2)).reshape(N_val, S_out, 2)
    # y_val_raw is (N_val, S_out, 2) [LAT, LON]
    
    pred_lat = y_pred_unscaled[:, :, 0]
    pred_lon = y_pred_unscaled[:, :, 1]
    true_lat = y_val_raw[:, :, 0]
    true_lon = y_val_raw[:, :, 1]
    
    # Latitude & Longitude MAE in degrees
    lat_mae = float(np.mean(np.abs(pred_lat - true_lat)))
    lon_mae = float(np.mean(np.abs(pred_lon - true_lon)))
    
    # Haversine distance in kilometers
    dist_matrix_km = haversine_distance_km(true_lat, true_lon, pred_lat, pred_lon)
    mean_dist_km = float(np.mean(dist_matrix_km))
    median_dist_km = float(np.median(dist_matrix_km))
    
    # Step-by-step horizon errors (Step 1 to Step 6)
    step_errors_km = {}
    for step in range(S_out):
        step_km = float(np.mean(dist_matrix_km[:, step]))
        step_errors_km[f"step_{step+1}"] = round(step_km, 2)
        
    report = {
        "model_file": str(MODEL_PATH),
        "validation_cyclones_count": len(val_sids),
        "validation_sequences_count": len(X_val_scaled),
        "validation_loss_scaled_mse": round(val_loss, 6),
        "latitude_mae_deg": round(lat_mae, 4),
        "longitude_mae_deg": round(lon_mae, 4),
        "overall_mean_position_error_km": round(mean_dist_km, 2),
        "overall_median_position_error_km": round(median_dist_km, 2),
        "horizon_step_errors_km": step_errors_km
    }
    
    report_file = REPORTS_DIR / "validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*65)
    print("CYCLOVISION AI — TRAJECTORY MODEL VALIDATION REPORT")
    print("="*65)
    print(f"Validation Cyclones (Unseen):    {report['validation_cyclones_count']}")
    print(f"Validation Sequences Evaluated: {report['validation_sequences_count']}")
    print(f"Validation Loss (Scaled MSE):   {report['validation_loss_scaled_mse']}")
    print(f"Latitude MAE:                   {report['latitude_mae_deg']}°")
    print(f"Longitude MAE:                  {report['longitude_mae_deg']}°")
    print(f"Overall Mean Position Error:    {report['overall_mean_position_error_km']} km")
    print(f"Overall Median Position Error:  {report['overall_median_position_error_km']} km")
    print("Forecast Horizon Error Breakdown:")
    for step, err in step_errors_km.items():
        print(f"  - {step.upper()}: {err:6.1f} km")
    print("="*65 + "\n")
    
    return report


if __name__ == "__main__":
    validate_trajectory_model()
