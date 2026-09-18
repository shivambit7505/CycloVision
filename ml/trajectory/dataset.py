"""
CycloVision AI - Trajectory Prediction Dataset Preprocessing Pipeline
Extracts temporal sliding sequences (PAST 6 observations -> FUTURE 6 observations)
from NOAA IBTrACS historical cyclone tracks.
Splits strictly by Cyclone SID to prevent data leakage.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/ibtracs/NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv"
OUTPUT_DIR = "models/trajectory"

INPUT_SEQ_LEN = 6   # Past 6 observations
OUTPUT_SEQ_LEN = 6  # Future 6 observations
TOTAL_WINDOW = INPUT_SEQ_LEN + OUTPUT_SEQ_LEN  # 12 points minimum per track


def clean_ibtracs_data(csv_path: str = DATA_PATH):
    """Loads and standardizes NOAA IBTrACS data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"IBTrACS dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Standardize coordinate and time types
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    
    # Filter valid coordinates
    df = df.dropna(subset=['LAT', 'LON', 'ISO_TIME', 'SID'])
    df = df[df['LAT'].between(-90, 90) & df['LON'].between(-180, 180)]
    
    # Combine wind from WMO, USA, and NEWDELHI
    df['WIND'] = (
        pd.to_numeric(df['WMO_WIND'], errors='coerce')
        .combine_first(pd.to_numeric(df['USA_WIND'], errors='coerce'))
        .combine_first(pd.to_numeric(df['NEWDELHI_WIND'], errors='coerce'))
    )
    
    # Combine pressure from WMO, USA, and NEWDELHI
    df['PRES'] = (
        pd.to_numeric(df['WMO_PRES'], errors='coerce')
        .combine_first(pd.to_numeric(df['USA_PRES'], errors='coerce'))
        .combine_first(pd.to_numeric(df['NEWDELHI_PRES'], errors='coerce'))
    )
    
    # Sort chronologically within each storm
    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)
    
    # Compute within-storm velocity / deltas
    df['DELTA_LAT'] = df.groupby('SID')['LAT'].diff().fillna(0.0)
    df['DELTA_LON'] = df.groupby('SID')['LON'].diff().fillna(0.0)
    
    # Impute missing wind/pres per storm using forward/backward fill, then global median
    df['WIND'] = df.groupby('SID')['WIND'].transform(lambda g: g.ffill().bfill())
    df['WIND'] = df['WIND'].fillna(df['WIND'].median() if df['WIND'].notnull().any() else 35.0)
    
    df['PRES'] = df.groupby('SID')['PRES'].transform(lambda g: g.ffill().bfill())
    df['PRES'] = df['PRES'].fillna(df['PRES'].median() if df['PRES'].notnull().any() else 1000.0)
    
    # Seasonal cyclical features (cyclones peak in pre-monsoon May & post-monsoon Oct/Nov)
    month = df['ISO_TIME'].dt.month
    df['MONTH_SIN'] = np.sin(2 * np.pi * month / 12.0)
    df['MONTH_COS'] = np.cos(2 * np.pi * month / 12.0)
    
    return df


def generate_sequences(val_split: float = 0.20, random_seed: int = 42):
    """
    Builds sliding window sequences partitioned by Cyclone SID.
    
    Returns:
        X_train, y_train, X_val, y_val, feature_scaler, target_scaler, train_sids, val_sids
    """
    df = clean_ibtracs_data()
    
    # Filter storms with at least TOTAL_WINDOW observations
    storm_sizes = df.groupby('SID').size()
    eligible_storms = storm_sizes[storm_sizes >= TOTAL_WINDOW].index.tolist()
    
    np.random.seed(random_seed)
    shuffled_sids = np.random.permutation(eligible_storms)
    
    val_size = int(len(shuffled_sids) * val_split)
    val_sids = set(shuffled_sids[:val_size])
    train_sids = set(shuffled_sids[val_size:])
    
    features = ['LAT', 'LON', 'DELTA_LAT', 'DELTA_LON', 'WIND', 'PRES', 'MONTH_SIN', 'MONTH_COS']
    target_cols = ['LAT', 'LON']
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    for sid, group in df.groupby('SID'):
        if sid not in eligible_storms:
            continue
            
        group_feats = group[features].values
        group_targets = group[target_cols].values
        n_obs = len(group)
        
        for i in range(n_obs - TOTAL_WINDOW + 1):
            x_seq = group_feats[i : i + INPUT_SEQ_LEN]
            y_seq = group_targets[i + INPUT_SEQ_LEN : i + TOTAL_WINDOW]
            
            if sid in val_sids:
                X_val_list.append(x_seq)
                y_val_list.append(y_seq)
            else:
                X_train_list.append(x_seq)
                y_train_list.append(y_seq)
                
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    
    # Fit scalers only on training split
    N_tr, S_in, F_in = X_train.shape
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, F_in)).reshape(N_tr, S_in, F_in)
    
    N_val, _, _ = X_val.shape
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, F_in)).reshape(N_val, S_in, F_in)
    
    # Target scaler for coordinates [LAT, LON]
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 2)).reshape(N_tr, OUTPUT_SEQ_LEN, 2)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 2)).reshape(N_val, OUTPUT_SEQ_LEN, 2)
    
    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "features": features,
        "target_cols": target_cols,
        "input_seq_len": INPUT_SEQ_LEN,
        "output_seq_len": OUTPUT_SEQ_LEN
    }
    
    return (
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled,
        y_train, y_val, # raw targets for unscaled metric evaluation
        scalers,
        list(train_sids), list(val_sids)
    )


if __name__ == "__main__":
    print("Testing sequence generator...")
    X_tr, y_tr, X_va, y_va, _, _, scalers, tr_sids, va_sids = generate_sequences()
    print(f"Train Cyclones:   {len(tr_sids)}")
    print(f"Val Cyclones:     {len(va_sids)}")
    print(f"X_train Shape:    {X_tr.shape}")
    print(f"y_train Shape:    {y_tr.shape}")
    print(f"X_val Shape:      {X_va.shape}")
    print(f"y_val Shape:      {y_va.shape}")
    print(f"Features:         {scalers['features']}")
