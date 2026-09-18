# -*- coding: utf-8 -*-
"""
CycloVision AI - NOAA IBTrACS Trajectory Dataset Preprocessing Pipeline
Extracts temporal sliding sequences (PAST 6 observations -> FUTURE 6 observations)
from official NOAA IBTrACS North Indian Ocean historical cyclone tracks.
Strictly partitions by Cyclone SID (70% Train, 15% Val, 15% Test) to ensure zero data leakage.
Fits StandardScalers exclusively on training data.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/ibtracs/NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv"
OUTPUT_DIR = "models/trajectory"

INPUT_SEQ_LEN = 6   # Past 6 observations
OUTPUT_SEQ_LEN = 6  # Future 6 observations
TOTAL_WINDOW = INPUT_SEQ_LEN + OUTPUT_SEQ_LEN  # 12 observations per sequence
MAX_STEP_HOURS = 6.0  # Maximum allowed continuous gap between tracking points


def clean_ibtracs_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads and cleans official NOAA IBTrACS North Indian Ocean dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IBTrACS dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # 1. Coordinate & timestamp validation
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')

    # Remove invalid records and drop duplicates on (SID, ISO_TIME)
    df = df.dropna(subset=['LAT', 'LON', 'ISO_TIME', 'SID'])
    df = df[df['LAT'].between(-90, 90) & df['LON'].between(-180, 180)]
    df = df.drop_duplicates(subset=['SID', 'ISO_TIME'])

    # Ensure NAME string
    df['NAME'] = df['NAME'].fillna('UNNAMED').astype(str)

    # 2. Wind fallback logic (WMO -> USA -> NEWDELHI -> BOM)
    df['WIND'] = (
        pd.to_numeric(df['WMO_WIND'], errors='coerce')
        .combine_first(pd.to_numeric(df['USA_WIND'], errors='coerce'))
        .combine_first(pd.to_numeric(df['NEWDELHI_WIND'], errors='coerce'))
        .combine_first(pd.to_numeric(df['BOM_WIND'], errors='coerce'))
    )

    # 3. Pressure fallback logic (WMO -> USA -> NEWDELHI -> BOM)
    df['PRES'] = (
        pd.to_numeric(df['WMO_PRES'], errors='coerce')
        .combine_first(pd.to_numeric(df['USA_PRES'], errors='coerce'))
        .combine_first(pd.to_numeric(df['NEWDELHI_PRES'], errors='coerce'))
        .combine_first(pd.to_numeric(df['BOM_PRES'], errors='coerce'))
    )

    # Sort chronologically by storm SID and timestamp
    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)

    # 4. Intra-cyclone velocity deltas
    df['DELTA_LAT'] = df.groupby('SID')['LAT'].diff().fillna(0.0)
    df['DELTA_LON'] = df.groupby('SID')['LON'].diff().fillna(0.0)

    # Impute missing wind / pressure per storm track, fallback to global median
    df['WIND'] = df.groupby('SID')['WIND'].transform(lambda g: g.ffill().bfill())
    df['WIND'] = df['WIND'].fillna(df['WIND'].median() if df['WIND'].notnull().any() else 35.0)

    df['PRES'] = df.groupby('SID')['PRES'].transform(lambda g: g.ffill().bfill())
    df['PRES'] = df['PRES'].fillna(df['PRES'].median() if df['PRES'].notnull().any() else 1005.0)

    # 5. Seasonal cyclical features
    month = df['ISO_TIME'].dt.month
    df['MONTH_SIN'] = np.sin(2 * np.pi * month / 12.0)
    df['MONTH_COS'] = np.cos(2 * np.pi * month / 12.0)

    return df


def generate_sequences(
    csv_path: str = DATA_PATH,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Builds sliding window sequences partitioned strictly by Cyclone SID (70% Train, 15% Val, 15% Test).
    Ensures zero data leakage between splits.
    Fits StandardScaler exclusively on training split.
    """
    df = clean_ibtracs_data(csv_path)

    # Identify eligible cyclones with >= TOTAL_WINDOW observations
    storm_sizes = df.groupby('SID').size()
    eligible_sids = sorted(storm_sizes[storm_sizes >= TOTAL_WINDOW].index.tolist())

    # Split SIDs deterministically
    np.random.seed(random_seed)
    shuffled_sids = np.random.permutation(eligible_sids)
    n_total = len(shuffled_sids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_sids = sorted(shuffled_sids[:n_train].tolist())
    val_sids = sorted(shuffled_sids[n_train : n_train + n_val].tolist())
    test_sids = sorted(shuffled_sids[n_train + n_val :].tolist())

    train_set = set(train_sids)
    val_set = set(val_sids)
    test_set = set(test_sids)

    # Strict disjointness verification
    assert len(train_set & val_set) == 0, "Train and Val SIDs overlap!"
    assert len(train_set & test_set) == 0, "Train and Test SIDs overlap!"
    assert len(val_set & test_set) == 0, "Val and Test SIDs overlap!"

    features = ['LAT', 'LON', 'DELTA_LAT', 'DELTA_LON', 'WIND', 'PRES', 'MONTH_SIN', 'MONTH_COS']
    target_cols = ['LAT', 'LON']

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    test_meta_list = []

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
            # Verify continuous tracking (intervals <= 6h)
            time_diffs_h = (window_times[1:] - window_times[:-1]).astype('timedelta64[s]').astype(float) / 3600.0
            if (time_diffs_h > MAX_STEP_HOURS).any():
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
                test_meta_list.append({
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

    # Fit StandardScaler ONLY on the training split
    N_tr, S_in, F_in = X_train.shape
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, F_in)).reshape(N_tr, S_in, F_in)

    N_val = len(X_val)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, F_in)).reshape(N_val, S_in, F_in)

    N_test = len(X_test)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, F_in)).reshape(N_test, S_in, F_in)

    # Target scaler for coordinates [LAT, LON] fitted ONLY on training targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 2)).reshape(N_tr, OUTPUT_SEQ_LEN, 2)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 2)).reshape(N_val, OUTPUT_SEQ_LEN, 2)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 2)).reshape(N_test, OUTPUT_SEQ_LEN, 2)

    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "target_cols": target_cols,
        "input_seq_len": INPUT_SEQ_LEN,
        "output_seq_len": OUTPUT_SEQ_LEN
    }

    return (
        X_train_scaled, y_train_scaled, y_train,
        X_val_scaled, y_val_scaled, y_val,
        X_test_scaled, y_test_scaled, y_test,
        test_meta_list,
        scalers,
        train_sids, val_sids, test_sids
    )


if __name__ == "__main__":
    print("Testing sequence generator with official NOAA IBTrACS dataset...")
    (
        X_tr, y_tr, y_tr_raw,
        X_va, y_va, y_va_raw,
        X_te, y_te, y_te_raw,
        meta_te,
        scalers,
        tr_sids, va_sids, te_sids
    ) = generate_sequences()

    print(f"Train Cyclones:   {len(tr_sids)} SIDs -> {len(X_tr)} sequences")
    print(f"Val Cyclones:     {len(va_sids)} SIDs -> {len(X_va)} sequences")
    print(f"Test Cyclones:    {len(te_sids)} SIDs -> {len(X_te)} sequences")
    print(f"X_train Shape:    {X_tr.shape}")
    print(f"y_train Shape:    {y_tr.shape}")
    print(f"X_test Shape:     {X_te.shape}")
    print(f"y_test Shape:     {y_te.shape}")
