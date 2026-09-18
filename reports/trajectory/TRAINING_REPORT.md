# 🌪️ CycloVision AI — Cyclone Trajectory Prediction Model Training Report

**Model Type:** 2-Layer Recurrent Neural Network (Gated Recurrent Unit - GRU)  
**Task:** Multi-Step Spatiotemporal Cyclone Trajectory Forecasting  
**Dataset:** NOAA International Best Track Archive for Climate Stewardship (IBTrACS) — North Indian Ocean Basin  
**Audit & Training Date:** September 19, 2026  

---

## 1. Dataset Provenance & Empirical Audit

- **Dataset Source:** NOAA IBTrACS NI Cyclone Track ML Dataset (`data/ibtracs/NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv`)
- **Total Historical Observations:** **57,841 records**
- **Total Columns:** **62 columns**
- **Total Historical Storm Tracks (by SID):** **1,858 unique cyclones**
- **Date Range:** **1842-10-25 03:00:00 to 2025-12-02 18:00:00** (183 years of historical storm tracks)
- **Coordinate Integrity:** 57,841 / 57,841 (100.00% valid Earth coordinates)
  - Latitude Range: `0.70°N` to `83.00°N`
  - Longitude Range: `30.20°E` to `100.00°E`
- **Wind Speed Availability:** Modern era records (>= 1980) have 93.7% combined coverage (`WMO_WIND`, `USA_WIND`, `NEWDELHI_WIND`).
- **Central Pressure Availability:** Modern era records have 68.4% combined coverage (`WMO_PRES`, `USA_PRES`, `NEWDELHI_PRES`).
- **Data Integrity Assurance:** No synthetic observations were fabricated. Only actual consecutive observations from genuine cyclone tracks with $\ge 12$ recorded points were used to generate temporal sequences.

---

## 2. Temporal Sequence Preprocessing Pipeline

- **Sequence Window Structure:**
  - **Input History:** PAST 6 consecutive observations ($t-5$ to $t$, corresponding to approximately 18 to 36 hours of trajectory history).
  - **Forecast Horizon:** FUTURE 6 consecutive positions ($t+1$ to $t+6$, corresponding to approximately 18 to 36 hours of future forecast).
- **Split Strategy:** Strict partition by **Cyclone Storm ID (`SID`)**. Zero observations from the same cyclone exist in both training and validation splits.
  - **Training Split:** **1,260 cyclone tracks** $\rightarrow$ **31,314 sliding window sequences** (80.8%)
  - **Validation Split:** **314 unseen cyclone tracks** $\rightarrow$ **7,427 sliding window sequences** (19.2%)
- **Input Feature Vector (8 features per step):**
  1. `LAT`: Current latitude (degrees North)
  2. `LON`: Current longitude (degrees East)
  3. `DELTA_LAT`: Instantaneous velocity in latitude ($\text{lat}_t - \text{lat}_{t-1}$)
  4. `DELTA_LON`: Instantaneous velocity in longitude ($\text{lon}_t - \text{lon}_{t-1}$)
  5. `WIND`: Maximum sustained wind speed (knots)
  6. `PRES`: Minimum central surface pressure (hPa / mb)
  7. `MONTH_SIN`: Cyclical seasonal representation ($\sin(2\pi \cdot \text{month} / 12)$)
  8. `MONTH_COS`: Cyclical seasonal representation ($\cos(2\pi \cdot \text{month} / 12)$)
- **Feature & Target Normalization:** Scikit-Learn `StandardScaler` fitted exclusively on the 31,314 training sequences. Serialized to [`models/trajectory/scaler.pkl`](file:///e:/CycloVision/models/trajectory/scaler.pkl).

---

## 3. Model Architecture & Training Configuration

- **Framework:** Keras 3.15.1 with PyTorch 2.11.0 Engine (`KERAS_BACKEND=torch`)
- **Hardware Profile:** Intel Core i5-12450HX CPU (8 worker threads)
- **Model Topology:**
  - `Input`: Shape `(batch_size, 6, 8)`
  - `GRU Layer 1`: 64 units, `return_sequences=True`
  - `Dropout`: 0.2 rate
  - `GRU Layer 2`: 32 units, `return_sequences=False`
  - `Dense Projection`: 32 units, ReLU activation
  - `Output Dense`: 12 units (6 future time steps $\times$ 2 coordinates)
  - `Reshape`: Shape `(batch_size, 6, 2)` for direct `[latitude, longitude]` prediction
- **Optimizer:** Adam (Initial learning rate: $1 \times 10^{-3}$, reduced to $1.25 \times 10^{-4}$ via `ReduceLROnPlateau`)
- **Loss Function:** Mean Squared Error (MSE) on normalized coordinates
- **Batch Size:** 64
- **Epochs Completed:** 15 epochs (Early stopping restored best weights from Epoch 14)
- **Total Training Duration:** **83.43 seconds** (1.39 minutes)

---

## 4. Actual Measured Validation Metrics

Evaluated on **7,427 sequences across 314 completely unseen historical cyclones**:

| Metric | Measured Value | Unit / Description |
| :--- | :---: | :--- |
| **Validation Loss (MSE)** | `0.004533` | Scaled coordinate MSE |
| **Validation MAE (Scaled)** | `0.0433` | Scaled coordinate MAE |
| **Latitude MAE** | **0.2621°** | Unscaled physical latitude error (~29.1 km) |
| **Longitude MAE** | **0.3814°** | Unscaled physical longitude error (~42.4 km) |
| **Overall Mean Position Error** | **54.71 km** | Great-circle Haversine distance across all 6 steps |
| **Overall Median Position Error**| **37.84 km** | Robust non-parametric position displacement |

### Step-by-Step Forecast Horizon Error Breakdown:
- **Step 1 (+3h to +6h):** **21.5 km** mean error
- **Step 2 (+6h to +12h):** **29.0 km** mean error
- **Step 3 (+9h to +18h):** **43.4 km** mean error
- **Step 4 (+12h to +24h):** **60.1 km** mean error
- **Step 5 (+15h to +30h):** **77.7 km** mean error
- **Step 6 (+18h to +36h):** **96.6 km** mean error

---

## 5. Inference Engine Verification (`ml/trajectory/predict.py`)

The inference engine was verified using CLI and Python function interfaces:
- Correctly loads `models/trajectory/trajectory_model.keras` and `models/trajectory/scaler.pkl`.
- Handles arbitrary observation sequences with automatic padding/clamping.
- Enforces Earth coordinate clamping: Latitude $[-90.0, 90.0]$, Longitude $[-180.0, 180.0]$.

### Real Validation Cyclone Test Case: Cyclone FANI (`SID: 2019116N02090`)
Input: 6 historical observations during early intensification in the Bay of Bengal (April 27, 2019).  
Output saved to [`reports/trajectory/test_prediction.json`](file:///e:/CycloVision/reports/trajectory/test_prediction.json):

```json
[
  { "step": 1, "predicted": [6.88°N, 88.65°E], "actual": [6.60°N, 88.50°E], "error": "35.0 km" },
  { "step": 2, "predicted": [7.13°N, 88.37°E], "actual": [6.80°N, 88.20°E], "error": "41.3 km" },
  { "step": 3, "predicted": [7.57°N, 88.20°E], "actual": [7.10°N, 88.10°E], "error": "53.1 km" },
  { "step": 4, "predicted": [7.91°N, 88.09°E], "actual": [7.20°N, 88.10°E], "error": "78.6 km" },
  { "step": 5, "predicted": [8.21°N, 88.07°E], "actual": [7.40°N, 88.10°E], "error": "89.8 km" },
  { "step": 6, "predicted": [8.56°N, 87.89°E], "actual": [7.60°N, 87.90°E], "error": "106.3 km" }
]
```
- **Cyclone FANI Mean Forecast Error (6 steps):** **67.35 km**

---

## 6. Persisted Artifacts on Disk

| Artifact Name | Path | Description |
| :--- | :--- | :--- |
| **Trained GRU Model** | [`models/trajectory/trajectory_model.keras`](file:///e:/CycloVision/models/trajectory/trajectory_model.keras) | Keras 3 saved model format |
| **Fitted Preprocessing Scaler** | [`models/trajectory/scaler.pkl`](file:///e:/CycloVision/models/trajectory/scaler.pkl) | Serialized feature & target StandardScaler |
| **Model Metadata & Config** | [`models/trajectory/config.json`](file:///e:/CycloVision/models/trajectory/config.json) | Hyperparameters, feature schema, training loss |
| **Inference CLI Script** | [`ml/trajectory/predict.py`](file:///e:/CycloVision/ml/trajectory/predict.py) | Standalone & importable inference engine |
| **Validation Evaluator** | [`ml/trajectory/validate.py`](file:///e:/CycloVision/ml/trajectory/validate.py) | Metric calculation on unseen cyclone tracks |
| **Training History** | [`reports/trajectory/training_history.csv`](file:///e:/CycloVision/reports/trajectory/training_history.csv) | Epoch-by-epoch loss and MAE metrics |
| **Validation Report** | [`reports/trajectory/validation_report.json`](file:///e:/CycloVision/reports/trajectory/validation_report.json) | Structured validation summary |
| **Test Case Prediction** | [`reports/trajectory/test_prediction.json`](file:///e:/CycloVision/reports/trajectory/test_prediction.json) | Historical vs predicted track for Cyclone FANI |

---

> [!NOTE]  
> **Disclaimer:** This model is designed as a rapid hackathon proof-of-concept prototype for multi-source AI fusion. It does not claim operational meteorology forecasting certification or replace official India Meteorological Department (IMD) or RSMC advisories.
