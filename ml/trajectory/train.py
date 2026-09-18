# -*- coding: utf-8 -*-
"""
CycloVision AI - Trajectory Prediction GRU Model Training
Trains a 2-layer recurrent neural network (GRU) sequence-to-vector regression model
on temporal sequences from the official NOAA IBTrACS dataset.
Saves model, scalers, train/val/test SIDs, and configuration for inference & evaluation.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Use PyTorch backend for Keras 3
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, callbacks
import torch

from ml.trajectory.dataset import generate_sequences

OUTPUT_DIR = Path("models/trajectory")
REPORTS_DIR = Path("reports/trajectory")


def build_trajectory_model(input_seq_len: int = 6, num_features: int = 8, output_seq_len: int = 6):
    """Builds a 2-layer GRU sequence-to-vector regression model."""
    inputs = layers.Input(shape=(input_seq_len, num_features), name="cyclone_sequence_input")

    # Recurrent feature extraction
    x = layers.GRU(64, return_sequences=True, name="gru_layer_1")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.GRU(32, return_sequences=False, name="gru_layer_2")(x)

    # Dense projection
    x = layers.Dense(32, activation="relu", name="dense_proj")(x)

    # Output: (output_seq_len * 2) coordinates
    outputs = layers.Dense(output_seq_len * 2, name="coord_dense")(x)
    outputs = layers.Reshape((output_seq_len, 2), name="lat_lon_output")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cyclovision_gru_trajectory")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "CUDA GPU" if torch.cuda.is_available() else f"CPU ({os.cpu_count() or 4} threads)"
    print("\n" + "=" * 65)
    print("CYCLOVISION AI — CYCLONE TRAJECTORY PREDICTION (GRU) TRAINING")
    print("=" * 65)
    print(f"Hardware Device: {device}")

    # Step 1: Preprocess and generate sequences (70% Train, 15% Val, 15% Test)
    print("[1/4] Loading and splitting NOAA IBTrACS dataset by Cyclone SID...")
    (
        X_train, y_train, y_train_raw,
        X_val, y_val, y_val_raw,
        X_test, y_test, y_test_raw,
        test_meta,
        scalers,
        train_sids, val_sids, test_sids
    ) = generate_sequences(random_seed=42)

    print(f"  Training sequences:   {len(X_train)} across {len(train_sids)} cyclone tracks (70%)")
    print(f"  Validation sequences: {len(X_val)} across {len(val_sids)} cyclone tracks (15%)")
    print(f"  Unseen Test sequences:{len(X_test)} across {len(test_sids)} cyclone tracks (15%)")
    print(f"  Input sequence:       {scalers['input_seq_len']} observations × {len(scalers['features'])} features")
    print(f"  Output forecast:      {scalers['output_seq_len']} future positions (LAT, LON)")

    # Step 2: Build model
    print("[2/4] Initializing 2-layer GRU architecture...")
    model = build_trajectory_model(
        input_seq_len=scalers['input_seq_len'],
        num_features=len(scalers['features']),
        output_seq_len=scalers['output_seq_len']
    )
    model.summary()

    # Callbacks
    model_save_path = OUTPUT_DIR / "trajectory_model.keras"
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]

    # Step 3: Train
    epochs = 12
    batch_size = 64
    print(f"[3/4] Training for up to {epochs} epochs (batch size: {batch_size})...")

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )
    train_duration = time.time() - start_time
    print(f"\n[COMPLETED] Training finished in {train_duration:.2f} seconds ({train_duration/60:.2f} mins).")

    # Step 4: Save Model, Scalers, SIDs, and Config
    print("[4/4] Saving model weights, scalers, and split partition files...")
    model.save(str(model_save_path))
    print(f"  Saved Keras model:        {model_save_path}")

    scaler_path = OUTPUT_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"  Saved Scaler:             {scaler_path}")

    # Save train, val, test SIDs
    with open(OUTPUT_DIR / "train_sids.json", "w") as f:
        json.dump(train_sids, f, indent=2)
    with open(OUTPUT_DIR / "val_sids.json", "w") as f:
        json.dump(val_sids, f, indent=2)
    with open(OUTPUT_DIR / "test_sids.json", "w") as f:
        json.dump(test_sids, f, indent=2)
    print(f"  Saved Split SIDs:         train_sids.json, val_sids.json, test_sids.json")

    final_tr_loss = float(history.history['loss'][-1])
    final_val_loss = float(history.history['val_loss'][-1])
    final_val_mae = float(history.history['val_mae'][-1])

    config = {
        "model_type": "GRU",
        "framework": f"Keras {keras.__version__} (PyTorch Backend)",
        "hardware": device,
        "input_seq_len": scalers['input_seq_len'],
        "output_seq_len": scalers['output_seq_len'],
        "num_features": len(scalers['features']),
        "features": scalers['features'],
        "target_cols": scalers['target_cols'],
        "train_cyclones_count": len(train_sids),
        "val_cyclones_count": len(val_sids),
        "test_cyclones_count": len(test_sids),
        "train_sequences_count": len(X_train),
        "val_sequences_count": len(X_val),
        "test_sequences_count": len(X_test),
        "training_time_seconds": round(train_duration, 2),
        "final_train_loss": round(final_tr_loss, 6),
        "final_val_loss": round(final_val_loss, 6),
        "final_val_mae": round(final_val_mae, 6),
        "model_file": str(model_save_path.name),
        "scaler_file": str(scaler_path.name)
    }

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved Config:             {config_path}")

    # Save training history CSV
    import pandas as pd
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(REPORTS_DIR / "training_history.csv", index_label="epoch")
    print(f"  Saved Training History:   {REPORTS_DIR / 'training_history.csv'}")

    print("\n" + "=" * 65)
    print(f"TRAINING SUMMARY: {train_duration:.2f}s | Train Loss: {final_tr_loss:.6f} | Val Loss: {final_val_loss:.6f}")
    print("=" * 65 + "\n")

    return {
        "training_time_seconds": round(train_duration, 2),
        "final_train_loss": round(final_tr_loss, 6),
        "final_val_loss": round(final_val_loss, 6),
        "train_sids_count": len(train_sids),
        "val_sids_count": len(val_sids),
        "test_sids_count": len(test_sids),
        "train_sequences_count": len(X_train),
        "val_sequences_count": len(X_val),
        "test_sequences_count": len(X_test)
    }


if __name__ == "__main__":
    train()
