# 🌀 CycloVision AI — Hackathon ML Model & System Readiness Audit

**Audit Date:** September 18, 2026  
**Target:** Internal Hackathon Prototype (8–9 Hours Remaining)  
**Focus:** Pure ground truth — identifying real implementations vs mocks, dataset locations, model training readiness, and exact blockers.

---

## 1. Ground Truth Audit of 16 Critical Checkpoints

### 1. NOAA IBTrACS Historical Cyclone Data Availability
- **Downloaded on Machine:** **YES** (`C:\Users\Shivam Kumar\Downloads\NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv`).
- **Available in Project Repository:** **NO**. The file was never imported into `E:\CycloVision\data\` or `E:\CycloVision\v2\data\`.
- **Repository Implementation:** `v2/data/ibtracs_loader.py` contains only an in-memory hardcoded dictionary (`HISTORICAL_STORMS_DATABASE`) with 3 hand-typed storms.

### 2. Exact Dataset Files, Records, Columns & Date Range
- **File:** `C:\Users\Shivam Kumar\Downloads\NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv` (12.1 MB)
- **Total Records:** **57,841 rows**
- **Columns (62 total):** `SID`, `SEASON`, `NUMBER`, `BASIN`, `SUBBASIN`, `NAME`, `ISO_TIME`, `DATE_UTC`, `TIME_UTC`, `NATURE`, `LAT`, `LON`, `WMO_WIND`, `WMO_PRES`, `WMO_AGENCY`, `TRACK_TYPE`, `DIST2LAND`, `LANDFALL`, `USA_ATCF_ID`, etc.
- **Date Range:** **1842-10-25 03:00:00 to 2025-12-02 18:00:00** (183 years of historical North Indian Ocean cyclone tracks).

### 3. Satellite-Image Dataset Locally
- **Real Scientific Satellite Imagery:** **NONE**. There are zero raw L1B/L2 NetCDF/HDF5 satellite files in `v2/data/raw/` or `static/uploads/`.
- **Synthetic/Calibrated Satellite Imagery:** **YES** — 90 generated multi-spectral infrared/water-vapor crops located in `datasets/cyclone_detection/` (train: 64, val: 16, test: 10) + 3 static demo PNG crops in `static/sample_cyclones/`.

### 4. YOLO Annotation / Label Files
- **Status:** **EXISTS & VERIFIED**.
- **Location:** `datasets/cyclone_detection/{train,val,test}/labels/`
- **Total Label Files:** 90 text files containing 156 bounding box annotations.
- **Validation:** 100% compliant with normalized YOLO format `[class cx cy w h]`. Zero malformed or out-of-bound labels.

### 5. YOLO data.yaml Configuration
- **Status:** **EXISTS & VALID**.
- **Location:** `datasets/cyclone_detection/data.yaml`
- **Classes:**
  - `0: cyclone_eye` (Low-Level Circulation Center / Central Calm Eye)
  - `1: cyclone_body` (Central Dense Overcast & Primary Convective Spiral Vortex)
- **Paths:** Point correctly to relative `train/images`, `val/images`, and `test/images`.

### 6. YOLO Model Training Status
- **Status:** **BASELINE TRAINED**.
- Transfer learning completed from `yolov8n.pt` using `ml/detection/train.py` (8 epochs, CPU adaptive profile).
- **Validation Metrics:** Precision: 0.9699 (97.0%), Recall: 0.5000, F1: 0.6598, mAP@50: 0.9309 (93.1%), mAP@50-95: 0.8549 (85.5%).

### 7. Physical .pt YOLO Weights Files
- **Status:** **PRESENT ON DISK**.
  - `models/yolo/best.pt` (6.2 MB) — Fine-tuned best model checkpoint
  - `models/yolo/last.pt` (6.2 MB) — Final epoch checkpoint
  - `yolov8n.pt` (6.5 MB) — Ultralytics base pretrained model in root

### 8. LSTM / GRU Trajectory Training Code
- **Status:** **COMPLETELY MISSING**.
- There is no training code for LSTM, GRU, or ConvLSTM anywhere in the project.
- `notebooks/06_track.ipynb` is an empty 311-byte stub.

### 9. LSTM / GRU Model Training Status
- **Status:** **NOT TRAINED**.
- `v2/models/swin_convlstm.py` contains an un-trained PyTorch `SwinConvLSTMPredictor` class definition. At runtime, the class bypasses neural inference entirely and calculates future storm positions using a trigonometric kinematic formula with Coriolis bias (`d_lat = dist_km * cos(rad) / 111.0`).

### 10. Open-Meteo Weather Integration
- **Status:** **100% REAL IMPLEMENTATION**.
- `v2/services/weather_service.py` connects live to `https://api.open-meteo.com/v1/forecast` via HTTP with timeout, exponential backoff retries, in-memory TTL caching, and local SQLite observation persistence.
- Verified live response: 27.0°C, 1008.8 hPa pressure, 4.7 km/h wind speed.

### 11. Backend API Endpoints: Real vs Mock / Demo
| Endpoint | Implementation Type | Description |
| :--- | :---: | :--- |
| `GET /` | **REAL** | Serves static operational dashboard (`static/v2_dashboard.html`). |
| `POST /api/satellite/upload` | **REAL** | Validates image, preserves original, resizes to 256×256, computes normalization statistics, runs YOLOv8 model (`models/yolo/best.pt`), stores metadata in SQLite WAL. |
| `GET /api/satellite/images` | **REAL** | Queries SQLite `satellite_images` table. |
| `GET /api/satellite/sources` | **REAL** | Dynamically checks local upload, NASA GIBS, and MOSDAC credentials. |
| `GET /api/weather/current` | **REAL** | Live Open-Meteo REST API call with SQLite observation caching. |
| `GET /api/weather/historical` | **REAL** | Live Open-Meteo historical archive REST API call. |
| `POST /api/location-risk` | **REAL MATH** | Exact Haversine great-circle distance to hardcoded municipal shelter coordinates. |
| `POST /api/what-if` | **EMPIRICAL MATH** | Linear thermodynamic sensitivity formula (`12*SST - 0.8*Shear + 0.5*Moisture`). |
| `POST /api/hitl/review` | **REAL** | Inserts review decisions into persistent SQLite WAL audit table. |
| `GET /api/audit-trail` | **REAL** | Reads persistent SQLite audit log. |
| `GET /api/v2/storms` | **MOCK / HARDCODED** | Returns in-memory dictionary with 3 storms (Fani, Amphan, Biparjoy). |
| `GET /api/v2/storm/{id}` | **HYBRID** | Uses 4-5 hardcoded track points; applies real Atkinson-Holliday wind inversion. |
| `GET /api/historical-replay/{id}`| **MOCK / HARDCODED** | Returns static 5-step JSON array for Cyclone Fani regardless of input. |
| `GET /api/model-verification` | **MOCK / HARDCODED** | Returns hardcoded benchmark metrics (`track_error_24h_km: 42.1`). |
| `POST /api/broadcast-whatsapp` | **MOCK** | Writes simulated alert to `latest_email_alert.txt`. |

### 12. Root Cause: Why Geospatial Decision Canvas / Map Was Blank
- **Root Cause Identified & Fixed:** A JavaScript syntax error occurred on lines 1248–1282 of `static/v2_dashboard.html`, where raw unescaped newlines were embedded inside single-quoted alert strings (`alert('...\n...')`).
- **Browser Impact:** When a syntax error occurs inside a `<script>` tag, the browser aborts execution of the entire script. As a result, `window.onload` never ran, and `initV2Map()` was never invoked, leaving the `#v2map` Leaflet canvas unmounted and completely blank.
- **Resolution:** Strings were properly escaped (`\n`), restoring Leaflet map mounting and tile loading.

### 13. Root Cause: Why Live Weather Observation Showed `--` Values
- **Root Cause Identified & Fixed:**
  1. The JavaScript syntax error noted in #12 halted the entire script before `fetchLiveWeather()` could be called.
  2. Inside `initV2Map()`, line 682 previously called `fetchOpenMeteoTelemetry()` (an undefined function name) instead of `fetchLiveWeather()`.
- **Resolution:** Replaced with `fetchLiveWeather(19.8, 85.8)`, which immediately updates `#quickWeatherTemp`, `#quickWeatherPres`, and `#quickWeatherWind` with real Open-Meteo telemetry on page load.

### 14. YOLOv8 Satellite LLCC Tab: Real Inference vs UI Mock
- **Status:** **REAL NEURAL INFERENCE PIPELINE**.
- The backend executes `validate_and_process_satellite_image()`, passes the 256×256 normalized crop to `models/yolo/best.pt`, runs YOLOv8 forward prediction, extracts bounding boxes, calculates circulation center and Dvorak T-numbers, and stores the image record in SQLite.

### 15. Historical Replay Data Source
- **Status:** **HARDCODED DEMO PLACEHOLDER**.
- The frontend uses an in-memory 5-step JavaScript array (`replayData`), and the backend endpoint `/api/historical-replay/{storm_id}` returns a static 5-step JSON array. It is not connected to any historical database.

### 16. GPU / CUDA Availability
- **Status:** **CPU ONLY (`torch.cuda.is_available() = False`)**.
- Hardware: Intel 12th Gen Core i5-12450HX (8 threads).
- CUDA drivers and GPU PyTorch builds are not available in the current environment.

---

## A. READY
1. **YOLOv8 Cyclone Detection Model Pipeline:**
   - Training script: `ml/detection/train.py`
   - Validation script: `ml/detection/validate.py`
   - Inference script: `ml/detection/predict.py`
   - Trained weights: `models/yolo/best.pt` (6.2 MB)
   - Baseline dataset: `datasets/cyclone_detection/` (90 images, 156 bounding boxes, `data.yaml`)
2. **Open-Meteo Weather Service:**
   - Pluggable provider `v2/services/weather_service.py` with caching and SQLite WAL persistence.
3. **Satellite Upload & Ingestion Engine:**
   - `POST /api/satellite/upload` with PIL validation, 256×256 normalization, and SQLite metadata.
4. **Interactive Dashboard:**
   - Restored Leaflet map, live Open-Meteo weather cards, 3D vortex WebGL, HITL SQLite ledger.

---

## B. MISSING
1. **NOAA IBTrACS Ingestion Pipeline:** The 57,841-row CSV exists in `C:\Users\Shivam Kumar\Downloads\`, but is completely missing from the project repository.
2. **LSTM / GRU Trajectory Dataset & Training Script:** Zero training code, zero sequential dataset generator, zero recurrent model checkpoints.
3. **Operational Satellite Raw Rasters:** No raw INSAT-3D HDF5/NetCDF files in repository.

---

## C. BROKEN
1. **Historical Replay Multi-Storm Switching:** Selecting Amphan or Biparjoy in Historical Replay still plays the hardcoded 5 steps of Cyclone Fani.
2. **`v2/models/swin_convlstm.py`:** PyTorch `SwinConvLSTMPredictor` class is un-trained and unused; trajectory prediction uses hardcoded trigonometry.

---

## D. MOCK / PLACEHOLDER
1. `v2/data/ibtracs_loader.py` — 3-storm hardcoded Python dictionary (`HISTORICAL_STORMS_DATABASE`).
2. `/api/historical-replay/{storm_id}` — 5-step hardcoded JSON response.
3. `/api/model-verification` — Static comparison numbers (`track_error_24h_km: 42.1`).
4. `/api/data-status` — Static string labels describing external feeds.
5. Multi-channel broadcast dispatchers (WhatsApp/Email) — Simulate dispatch by writing to a local text file.

---

## E. DATASET LOCATION
| Dataset | Location | Record Count / Size | Status |
| :--- | :--- | :--- | :--- |
| **NOAA IBTrACS (Full)** | `C:\Users\Shivam Kumar\Downloads\NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv` | 57,841 rows, 62 columns (12.1 MB) | External Download (Needs ingestion) |
| **YOLO Cyclone Detection** | `E:\CycloVision\datasets\cyclone_detection\` | 90 images, 156 bounding boxes | In-Repo & Verified Valid |
| **Processed Tensors** | `E:\CycloVision\v2\data\processed\*.npz` | 3 files (Fani, Amphan, Biparjoy) | Synthetic 4×256×256 arrays |
| **Sample Satellite Crops** | `E:\CycloVision\static\sample_cyclones\` | 3 PNG images | In-Repo |

---

## F. YOLO TRAINING READINESS
- **Readiness:** **100% READY (ALREADY TRAINED & VERIFIED)**.
- Pretrained weights `yolov8n.pt` are present.
- Training script `ml/detection/train.py` is configured with CPU hardware adaptation (`imgsz=320`, `epochs=8`, `batch=8`).
- Baseline run produced `models/yolo/best.pt` with 93.1% mAP@50 and 22.12 ms inference speed.
- Can be re-trained or fine-tuned on demand at any time with a single command: `python ml/detection/train.py`.

---

## G. LSTM / GRU TRAINING READINESS
- **Readiness:** **NOT READY (0% READY)**.
- **Blockers:**
  1. The 57,841-row NOAA IBTrACS dataset is in `Downloads`, not in the repository.
  2. No sequence dataset generator exists to create 6h/12h/24h trajectory sliding windows `(lat_t, lon_t, wind_t, pres_t) -> (lat_{t+k}, lon_{t+k})`.
  3. No LSTM/GRU training script (`ml/trajectory/train_lstm.py`) exists.

---

## H. GPU STATUS
- **CUDA Availability:** **FALSE (CPU ONLY)**.
- **Hardware:** Intel Core i5-12450HX (8 threads).
- **Impact:** Model architectures must remain lightweight (YOLOv8-nano, small LSTM/GRU hidden dimensions `hidden_dim=32/64`) and image resolutions kept at 320×320 to avoid multi-hour training bottlenecks.

---

## I. EXACT NEXT STEP
To reach a fully functional hackathon prototype with **both** vision and sequence AI components:
1. Copy `C:\Users\Shivam Kumar\Downloads\NOAA_IBTrACS_NI_Cyclone_Track_ML_Dataset.csv` into `E:\CycloVision\data\ibtracs\`.
2. Build `ml/trajectory/dataset.py` to parse IBTrACS historical cyclone tracks into sequential trajectory time-series.
3. Build and train `ml/trajectory/train_lstm.py` (PyTorch GRU/LSTM for 6h–48h track forecasting) and save weights to `models/trajectory/best_lstm.pt`.
4. Connect `v2/models/swin_convlstm.py` to load and run inference with `best_lstm.pt`.

---

## CLEAR FINAL RECOMMENDATION

### **DO NOT TRAIN YET — FIX/CREATE: INGEST NOAA IBTrACS DATASET FROM DOWNLOADS AND CREATE THE LSTM/GRU TRAJECTORY TRAINING PIPELINE**

*(YOLO detection model has already been trained with `best.pt` saved; the single largest missing AI component is the LSTM/GRU trajectory training pipeline).*

