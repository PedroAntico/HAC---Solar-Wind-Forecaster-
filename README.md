# 🌪️ HAC — Solar Wind Forecaster  
### *Heliogeophysical Adaptive Coupling (HAC) Framework*

Scientific framework for analyzing adaptive Sun–Earth coupling using plasma physics, complex systems theory, and multifractal analysis to investigate heliogeophysical self-organization.

---

## 🚀 Overview

**HAC – Solar Wind Forecaster** is an AI-powered system designed to model and forecast solar wind parameters using real OMNI/NOAA data and advanced deep learning architectures.  
It blends **plasma physics**, **complex systems theory**, **adaptive coupling principles**, and **multifractal dynamics** into a unified scientific framework.

The system supports:

- **Real-time predictions** (1h → 48h)
- **Deep learning ensembles (LSTM, GRU, Hybrid Attention CNN-LSTM)**
- **Uncertainty quantification** (confidence intervals)
- **SHAP interpretability**
- **Solar wind feature engineering**
- **Adaptive coupling metrics**
- **Calibration curves and reliability analysis**
- **Interactive live dashboard (Dash/Plotly)**
- **REST API for forecasting**

---

## 📁 Project Structure

HAC-Solar-Wind-Forecaster/ 
├── config.yaml              # Global configuration 
├── hac_v6_train.py         # Training pipeline 
├── hac_v6_predictor.py     # Real-time predictor & API 
├── hac_v6_dashboard.py     # Interactive dashboard 
├── hac_v6_models.py        # Model architectures 
├── hac_v6_features.py      # Feature engineering + plasma features 
├── hac_v6_metrics.py       # Metrics, uncertainty, calibration 
├── hac_v6_config.py        # Config loader 
├── data_real/              # Real OMNI data 
├── models/                 # Stored models 
├── results/                # Results + plots 
└── docs/                   # Documentation

---

## 🔧 Installation

```bash
pip install -r requirements.txt

(Optional) Create venv:

python3 -m venv venv
source venv/bin/activate


---

🧠 Training the Models

python hac_v6_train.py

The framework will:

Load real OMNI data

Perform adaptive feature engineering

Train multi-horizon models

Generate SHAP interpretations

Compute uncertainty + confidence intervals

Save models + metadata



---

🔮 Real-Time Forecast API

Start service:

python hac_v6_predictor.py

Example request:

GET /api/v1/forecast?model_type=ensemble&horizon=24

Example response:

{
  "timestamp": "2025-01-15T12:00:00Z",
  "predictions": {
    "24": {
      "speed": 468.2,
      "bz_gse": -3.1,
      "density": 7.0
    }
  },
  "alerts": []
}


---

📊 Live Dashboard

python hac_v6_dashboard.py

Provides:

Forecast plots with uncertainty bands

Real-time alerts

Model performance maps

SHAP-based feature importance


Accessible via browser:

http://localhost:8050


---

👨‍🔬 Scientific Context

The HAC framework integrates:

Solar wind plasma dynamics

Heliospheric magnetic field fluctuations

Non-linear Sun–Earth coupling

Self-organization and emergent regimes

Multifractal turbulence signatures

Adaptive coupling indicators


Suitable for heliophysics research, space weather forecasting, and machine learning in geospace applications.


---

📄 License

MIT License


---

👤 Author

Pedro Antico
Heliogeophysical Adaptive Coupling Research Initiative
