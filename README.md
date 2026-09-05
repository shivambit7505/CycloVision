# CycloVision AI 🌀
### An AI/ML-Based System for Identification, Classification and Prediction of Tropical Cyclone Patterns Using Multi-Source Satellite Data

**Problem Statement ID:** SIH26070  
**Organization:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)  
**Category:** Software | **Theme:** Disaster Management  

---

## 🌟 Key Features
1. **Live Satellite Image Ingestion & Upload:** Supports raw INSAT-3D/3DR TIR & Visible imagery with automated Low-Level Circulation Center (LLCC) detection.
2. **Objective Dvorak Intensity Estimation:** Deep CNN trained on multi-spectral satellite imagery to estimate continuous T-numbers ($T1.0 - T8.0$) and Maximum Sustained Winds (MSW).
3. **Rapid Intensification (RI) Early Warning Engine:** Physics-informed thermodynamic risk engine evaluating Sea Surface Temperature (SST), Ocean Heat Content (OHC), and Vertical Wind Shear (VWS).
4. **Multi-Horizon Trajectory & Landfall Forecasting:** Multi-step ensemble path forecasting ($+6\text{h}$ to $+120\text{h}$) with dynamic Quantile Cones of Uncertainty.
5. **District-Level Coastal Vulnerability & Storm Surge Matrix:** Evaluates coastal districts, expected surge heights, and population exposure.
6. **Historical Cyclone Analog Similarity:** Compares active systems with historical IMD Best Track cyclones (*Super Cyclone 1999, Phailin, Fani, Amphan, Biparjoy*).
7. **Automated IMD Standard Bulletin Dispatcher:** Generates official government weather advisories in real-time.

---

## 🚀 Quickstart & Installation

```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd CycloVision-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (Weights cyclovision_weights.pth already included)
python train_dvorak.py

# 4. Start the FastAPI Command Center
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open your browser at:
👉 **`http://127.0.0.1:8000/static/index.html`**
