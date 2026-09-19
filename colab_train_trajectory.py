# -*- coding: utf-8 -*-
"""
CycloVision AI — Google Colab T4 GPU Trajectory Training & Evaluation Script
Directly executable in Google Colab environment.

Features:
1. Automatic GPU Detection (NVIDIA T4 / V100 / A100 or CPU fallback)
2. Loads official NOAA IBTrACS North Indian Ocean Dataset
3. 70/15/15 Cyclone-SID Split (Zero Data Leakage)
4. Fits StandardScaler strictly on training split
5. Trains 2-layer GRU Sequence-to-Vector Regression Architecture
6. Saves model weights, scalers, and train/val/test SIDs
7. Runs Final Unseen Test Evaluation across +6h to +36h horizons
8. Generates test_report.json, test_predictions.csv, and diagnostic charts
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Keras PyTorch backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, callbacks
import torch
from sklearn.preprocessing import StandardScaler

# Dataset and output paths
DATA_PATH = "data/ibtracs/NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv"
OUTPUT_DIR = Path("models/trajectory")
REPORTS_DIR = Path("reports/trajectory")

INPUT_SEQ_LEN = 6   # Past 6 observations
OUTPUT_SEQ_LEN = 6  # Future 6 observations
TOTAL_WINDOW = INPUT_SEQ_LEN + OUTPUT_SEQ_LEN  # 12 observations
MAX_STEP_HOURS = 6.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Computes great-circle distance between coordinates in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c


def clean_and_prepare_dataset(csv_path: str = DATA_PATH):
    print("\n[1/6] Loading and cleaning NOAA IBTrACS dataset...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please verify file path.")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw records loaded: {len(df)} rows across {len(df.columns)} columns")

    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')

    # Remove invalid coordinates and duplicate timestamps
    df = df.dropna(subset=['LAT', 'LON', 'ISO_TIME', 'SID'])
    df = df[df['LAT'].between(-90, 90) & df['LON'].between(-180, 180)]
    df = df.drop_duplicates(subset=['SID', 'ISO_TIME'])
    df['NAME'] = df['NAME'].fillna('UNNAMED').astype(str)

    # Fallback wind & pressure logic
    df['WIND'] = (
        pd.to_numeric(df['WMO_WIND'], errors='coerce')
        .combine_first(pd.to_numeric(df['USA_WIND'], errors='coerce'))
        .combine_first(pd.to_numeric(df['NEWDELHI_WIND'], errors='coerce'))
        .combine_first(pd.to_numeric(df['BOM_WIND'], errors='coerce'))
    )

    df['PRES'] = (
        pd.to_numeric(df['WMO_PRES'], errors='coerce')
        .combine_first(pd.to_numeric(df['USA_PRES'], errors='coerce'))
        .combine_first(pd.to_numeric(df['NEWDELHI_PRES'], errors='coerce'))
        .combine_first(pd.to_numeric(df['BOM_PRES'], errors='coerce'))
    )

    # Sort chronologically by SID and ISO_TIME
    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)

    # Velocity deltas
    df['DELTA_LAT'] = df.groupby('SID')['LAT'].diff().fillna(0.0)
    df['DELTA_LON'] = df.groupby('SID')['LON'].diff().fillna(0.0)

    # Impute missing values within each storm
    df['WIND'] = df.groupby('SID')['WIND'].transform(lambda g: g.ffill().bfill())
    df['WIND'] = df['WIND'].fillna(df['WIND'].median() if df['WIND'].notnull().any() else 35.0)

    df['PRES'] = df.groupby('SID')['PRES'].transform(lambda g: g.ffill().bfill())
    df['PRES'] = df['PRES'].fillna(df['PRES'].median() if df['PRES'].notnull().any() else 1005.0)

    # Cyclical seasonal features
    month = df['ISO_TIME'].dt.month
    df['MONTH_SIN'] = np.sin(2 * np.pi * month / 12.0)
    df['MONTH_COS'] = np.cos(2 * np.pi * month / 12.0)

    print(f"  Cleaned valid records: {len(df)} across {df['SID'].nunique()} unique cyclone SIDs.")
    return df


def split_and_create_sequences(df: pd.DataFrame, random_seed: int = 42):
    print("\n[2/6] Generating continuous sequences with 70/15/15 SID partition...")
    storm_sizes = df.groupby('SID').size()
    eligible_sids = sorted(storm_sizes[storm_sizes >= TOTAL_WINDOW].index.tolist())

    np.random.seed(random_seed)
    shuffled_sids = np.random.permutation(eligible_sids)
    n_total = len(shuffled_sids)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_sids = sorted(shuffled_sids[:n_train].tolist())
    val_sids = sorted(shuffled_sids[n_train : n_train + n_val].tolist())
    test_sids = sorted(shuffled_sids[n_train + n_val :].tolist())

    train_set, val_set, test_set = set(train_sids), set(val_sids), set(test_sids)
    assert len(train_set & val_set) == 0 and len(train_set & test_set) == 0 and len(val_set & test_set) == 0, "Data leakage detected!"

    features = ['LAT', 'LON', 'DELTA_LAT', 'DELTA_LON', 'WIND', 'PRES', 'MONTH_SIN', 'MONTH_COS']
    target_cols = ['LAT', 'LON']

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    meta_test_list = []

    for sid, group in df.groupby('SID'):
        if sid not in eligible_sids:
            continue

        group_feats = group[features].values
        group_targets = group[target_cols].values
        times = group['ISO_TIME'].values
        storm_name = str(group['NAME'].iloc[0])
        n_obs = len(group)

        for i in range(n_obs - TOTAL_WINDOW + 1):
            window_times = times[i : i + TOTAL_WINDOW]
            diffs_h = (window_times[1:] - window_times[:-1]).astype('timedelta64[s]').astype(float) / 3600.0
            if (diffs_h > MAX_STEP_HOURS).any():
                continue

            x_seq = group_feats[i : i + INPUT_SEQ_LEN]
            y_seq = group_targets[i + INPUT_SEQ_LEN : i + TOTAL_WINDOW]

            if sid in train_set:
                X_train_list.append(x_seq)
                y_train_list.append(y_seq)
            elif sid in val_set:
                X_val_list.append(x_seq)
                y_val_list.append(y_seq)
            elif sid in test_set:
                X_test_list.append(x_seq)
                y_test_list.append(y_seq)
                meta_test_list.append({
                    "sid": sid,
                    "storm_name": storm_name,
                    "initial_time": str(pd.to_datetime(window_times[INPUT_SEQ_LEN - 1])),
                    "forecast_times": [str(pd.to_datetime(t)) for t in window_times[INPUT_SEQ_LEN:]]
                })

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    # Fit StandardScalers ONLY on Training Data
    print("  Fitting StandardScaler strictly on training split (no leakage)...")
    N_tr, S_in, F_in = X_train.shape
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, F_in)).reshape(N_tr, S_in, F_in)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, F_in)).reshape(len(X_val), S_in, F_in)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, F_in)).reshape(len(X_test), S_in, F_in)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 2)).reshape(N_tr, OUTPUT_SEQ_LEN, 2)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 2)).reshape(len(y_val), OUTPUT_SEQ_LEN, 2)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 2)).reshape(len(y_test), OUTPUT_SEQ_LEN, 2)

    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "target_cols": target_cols,
        "input_seq_len": INPUT_SEQ_LEN,
        "output_seq_len": OUTPUT_SEQ_LEN
    }

    print(f"  Train: {len(train_sids)} SIDs -> {len(X_train)} sequences")
    print(f"  Val:   {len(val_sids)} SIDs -> {len(X_val)} sequences")
    print(f"  Test:  {len(test_sids)} SIDs -> {len(X_test)} sequences")

    return (
        X_train_scaled, y_train_scaled, y_train,
        X_val_scaled, y_val_scaled, y_val,
        X_test_scaled, y_test_scaled, y_test,
        meta_test_list,
        scalers,
        train_sids, val_sids, test_sids
    )


def build_model(input_seq_len: int = 6, num_features: int = 8, output_seq_len: int = 6):
    inputs = layers.Input(shape=(input_seq_len, num_features), name="cyclone_sequence_input")
    x = layers.GRU(64, return_sequences=True, name="gru_layer_1")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.GRU(32, return_sequences=False, name="gru_layer_2")(x)
    x = layers.Dense(32, activation="relu", name="dense_proj")(x)
    outputs = layers.Dense(output_seq_len * 2, name="coord_dense")(x)
    outputs = layers.Reshape((output_seq_len, 2), name="lat_lon_output")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cyclovision_gru_trajectory")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CYCLOVISION AI — GOOGLE COLAB TRAJECTORY MODEL TRAINING & TEST")
    print("=" * 65)

    # Hardware detection
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Hardware: CUDA GPU Detected -> {device_name} ({vram_gb:.2f} GB VRAM)")
        batch_size = 64
        epochs = 20
    else:
        print(f"Hardware: CPU Mode ({os.cpu_count()} threads).")
        batch_size = 64
        epochs = 12

    # 1. Clean data & create splits
    df = clean_and_prepare_dataset(DATA_PATH)
    (
        X_train, y_train, y_train_raw,
        X_val, y_val, y_val_raw,
        X_test, y_test, y_test_raw,
        meta_test,
        scalers,
        train_sids, val_sids, test_sids
    ) = split_and_create_sequences(df, random_seed=42)

    # 2. Build model
    print("\n[3/6] Initializing GRU sequence model...")
    model = build_model(
        input_seq_len=scalers['input_seq_len'],
        num_features=len(scalers['features']),
        output_seq_len=scalers['output_seq_len']
    )
    model.summary()

    # 3. Train model
    print(f"\n[4/6] Training on GPU/CPU for up to {epochs} epochs (batch size: {batch_size})...")
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )
    train_duration = time.time() - t0
    print(f"\nTraining completed in {train_duration:.2f}s ({train_duration/60:.2f} mins).")

    # 4. Save Model & Artifacts
    print("\n[5/6] Saving model, scalers, and SID partitions...")
    model_path = OUTPUT_DIR / "trajectory_model.keras"
    scaler_path = OUTPUT_DIR / "scaler.pkl"

    model.save(str(model_path))
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    with open(OUTPUT_DIR / "train_sids.json", "w") as f:
        json.dump(train_sids, f, indent=2)
    with open(OUTPUT_DIR / "val_sids.json", "w") as f:
        json.dump(val_sids, f, indent=2)
    with open(OUTPUT_DIR / "test_sids.json", "w") as f:
        json.dump(test_sids, f, indent=2)

    print(f"  Saved model: {model_path}")
    print(f"  Saved scaler: {scaler_path}")
    print(f"  Saved SIDs: train_sids.json, val_sids.json, test_sids.json")

    # 5. Final Unseen Test Evaluation
    print("\n[6/6] Running final evaluation on UNSEEN TEST CYCLONES...")
    target_scaler = scalers["target_scaler"]

    with torch.no_grad():
        preds_scaled = model.predict(X_test, batch_size=128, verbose=0)
        preds_unscaled = target_scaler.inverse_transform(preds_scaled.reshape(-1, 2)).reshape(len(X_test), OUTPUT_SEQ_LEN, 2)

    all_rows = []
    lat_errors, lon_errors = [], []
    step_errors = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    lead_map = {1: "6h", 2: "12h", 3: "18h", 4: "24h", 5: "30h", 6: "36h"}

    for i in range(len(X_test)):
        meta = meta_test[i]
        for s in range(1, OUTPUT_SEQ_LEN + 1):
            s_idx = s - 1
            act_lat, act_lon = float(y_test_raw[i, s_idx, 0]), float(y_test_raw[i, s_idx, 1])
            pred_lat, pred_lon = float(preds_unscaled[i, s_idx, 0]), float(preds_unscaled[i, s_idx, 1])

            pred_lat = round(max(-90.0, min(90.0, pred_lat)), 4)
            pred_lon = round(max(-180.0, min(180.0, pred_lon)), 4)
            err_km = float(haversine_km(act_lat, act_lon, pred_lat, pred_lon))

            lat_errors.append(abs(act_lat - pred_lat))
            lon_errors.append(abs(act_lon - pred_lon))
            step_errors[s].append(err_km)

            all_rows.append({
                "SID": meta["sid"],
                "storm_name": meta["storm_name"],
                "initial_time": meta["initial_time"],
                "forecast_step": s,
                "forecast_time": meta["forecast_times"][s_idx] if s_idx < len(meta["forecast_times"]) else "",
                "actual_lat": round(act_lat, 4),
                "actual_lon": round(act_lon, 4),
                "predicted_lat": pred_lat,
                "predicted_lon": pred_lon,
                "error_km": round(err_km, 2)
            })

    pred_df = pd.DataFrame(all_rows)
    pred_csv_path = REPORTS_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved test predictions: {pred_csv_path} ({len(pred_df)} rows)")

    mean_h = float(np.mean(pred_df['error_km']))
    median_h = float(np.median(pred_df['error_km']))
    lat_mae = float(np.mean(lat_errors))
    lon_mae = float(np.mean(lon_errors))

    step_means = {lead_map[s]: round(float(np.mean(step_errors[s])), 2) for s in range(1, 7)}
    step_medians = {lead_map[s]: round(float(np.median(step_errors[s])), 2) for s in range(1, 7)}

    test_report = {
        "evaluation_name": "Official NOAA IBTrACS Unseen Test Split Evaluation (Colab GPU)",
        "dataset_path": DATA_PATH,
        "model_file": str(model_path),
        "scaler_file": str(scaler_path),
        "total_test_cyclone_sids": len(test_sids),
        "total_test_sequences": len(X_test),
        "total_evaluated_forecast_points": len(all_rows),
        "zero_data_leakage_verified": True,
        "metrics": {
            "latitude_mae_deg": round(lat_mae, 4),
            "longitude_mae_deg": round(lon_mae, 4),
            "mean_haversine_error_km": round(mean_h, 2),
            "median_haversine_error_km": round(median_h, 2),
            "step_horizon_errors_km": {
                "6h_error": step_means["6h"],
                "12h_error": step_means["12h"],
                "18h_error": step_means["18h"],
                "24h_error": step_means["24h"],
                "30h_error": step_means["30h"],
                "36h_error": step_means["36h"]
            },
            "step_horizon_median_errors_km": {
                "6h_median": step_medians["6h"],
                "12h_median": step_medians["12h"],
                "18h_median": step_medians["18h"],
                "24h_median": step_medians["24h"],
                "30h_median": step_medians["30h"],
                "36h_median": step_medians["36h"]
            }
        }
    }

    report_json_path = REPORTS_DIR / "test_report.json"
    with open(report_json_path, "w") as f:
        json.dump(test_report, f, indent=2)
    print(f"  Saved test report: {report_json_path}")

    # Generate Visual Plots
    print("  Generating diagnostic charts...")
    horizons = ["6h", "12h", "18h", "24h", "30h", "36h"]
    means = [step_means[h] for h in horizons]
    medians = [step_medians[h] for h in horizons]

    plt.figure(figsize=(9, 5), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    xi = np.arange(len(horizons))
    w = 0.35
    b1 = ax.bar(xi - w/2, means, w, label='Mean Error (km)', color='#06b6d4', edgecolor='#38bdf8')
    b2 = ax.bar(xi + w/2, medians, w, label='Median Error (km)', color='#10b981', edgecolor='#34d399')
    ax.set_xlabel('Forecast Horizon', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax.set_ylabel('Position Error (km)', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax.set_title('CycloVision AI — GRU Forecast Horizon Error (Unseen Test Cyclones)', fontsize=12, fontweight='bold', color='#f8fafc')
    ax.set_xticks(xi)
    ax.set_xticklabels(horizons, color='#e2e8f0', fontweight='bold')
    ax.tick_params(colors='#e2e8f0')
    ax.grid(axis='y', linestyle='--', alpha=0.25, color='#cbd5e1')
    ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "horizon_error.png", dpi=150, facecolor=plt.gcf().get_facecolor())
    plt.close()

    plt.figure(figsize=(9, 6), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    sample_sids = pred_df['SID'].drop_duplicates().tolist()[:4]
    colors = ['#38bdf8', '#f43f5e', '#a855f7', '#fbbf24']
    for idx, s_id in enumerate(sample_sids):
        s_data = pred_df[pred_df['SID'] == s_id].iloc[:6]
        c = colors[idx % len(colors)]
        ax.plot(s_data['actual_lon'], s_data['actual_lat'], marker='o', linestyle='-', color=c, label=f'{s_id} Actual', linewidth=2)
        ax.plot(s_data['predicted_lon'], s_data['predicted_lat'], marker='x', linestyle='--', color=c, alpha=0.8, label=f'{s_id} GRU Pred', linewidth=1.8)
    ax.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold', color='#e2e8f0')
    ax.set_title('CycloVision AI — Track Comparison (Sample Unseen Test Cyclones)', fontsize=12, fontweight='bold', color='#f8fafc')
    ax.tick_params(colors='#e2e8f0')
    ax.grid(True, linestyle='--', alpha=0.25, color='#cbd5e1')
    ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "track_comparison.png", dpi=150, facecolor=plt.gcf().get_facecolor())
    plt.close()

    print("\n" + "=" * 65)
    print("FINAL UNSEEN TEST EVALUATION METRICS:")
    print(f"  Latitude MAE:            {lat_mae:.4f}°")
    print(f"  Longitude MAE:           {lon_mae:.4f}°")
    print(f"  Mean Haversine Error:    {mean_h:.2f} km")
    print(f"  Median Haversine Error:  {median_h:.2f} km")
    print("  Step-by-Step Horizon Errors:")
    for h in horizons:
        print(f"    +{h:<3}: Mean = {step_means[h]:>6.2f} km | Median = {step_medians[h]:>6.2f} km")
    print("=" * 65)
    print("\nCOLAB TRAINING & EVALUATION PIPELINE COMPLETE!")


if __name__ == "__main__":
    main()
