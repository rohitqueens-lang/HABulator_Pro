<div align="center">

# 🌊 Habulator

**Great Lakes Phytoplankton Biovolume Emulator**

*XGBoost · Conformal Prediction · SHAP Explanations*

[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Frontend: Vercel](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://vercel.com)
[![API: Render](https://img.shields.io/badge/API-Render-46E3B7?logo=render)](https://render.com)

</div>

---

## What is Habulator?

Habulator is a machine-learning emulator for predicting phytoplankton biovolume in the
**Laurentian Great Lakes (2001–2021)**. Given six environmental conditions, it returns:

- **Point prediction** in mg/L (Duan smearing–corrected)
- **90% prediction interval** via conformalized quantile regression (CQR)
- **Feature-level SHAP explanation** — why the model predicted that value

It covers five phytoplankton groups — **Early Diatoms, Late Diatoms, Chlorophytes,
Cryptophytes, Cyanobacteria** — all trained and served.

---

## Features

| Feature | Detail |
|---|---|
| 🎯 **Predictions** | XGBoost trained on 20+ years of U.S. EPA GLNPO monitoring data |
| 📊 **Uncertainty** | 90% prediction intervals via conformalized quantile regression |
| 🔍 **Explainability** | Per-prediction SHAP feature contributions |
| 🧪 **Reproducible** | Training pipeline, curated dataset, and figure scripts all included |
| 🔌 **Open API** | FastAPI with interactive OpenAPI docs at `/docs` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  User sets 6 environmental inputs                         │
│  (TEMP, TP, SI, NO23, Station Depth, Day of Year)         │
└───────────────────────┬──────────────────────────────────┘
                        │  POST /predict  {group, ...features}
                        ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI (Python)                                         │
│  ├─ Feature engineering → 8 model features                │
│  │    DOY → DOY_sin, DOY_cos   ·   NO23 / TP → NP_ratio   │
│  ├─ XGBoost on log(1 + biovolume)                         │
│  ├─ Duan smearing back-transform → mg/L                   │
│  ├─ CQR quantile bounds → 90% interval [low, high]        │
│  └─ SHAP TreeExplainer → per-feature contributions        │
└───────────────────────┬──────────────────────────────────┘
                        │  JSON
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Next.js frontend (Vercel)                               │
│  ├─ Point estimate  +  mg/L ↔ log toggle                  │
│  ├─ 90% interval gauge  (lower · estimate · upper)        │
│  └─ SHAP bars: Temperature ▲ +0.38 · Total P ▼ −0.18     │
└──────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
train_emulator.py            Training pipeline (Optuna HPO, CQR, smearing → artifacts)
build_master_dataset_v2.py   Builds habulator_master_v2.csv from the raw EPA sources
make_*.py                    Manuscript figure scripts (study area, performance, uncertainty, SHAP)
habulator_master_v2.csv      Curated dataset (2,316 samples, 73 stations)
*_metrics.json               Per-group training metrics
figures/                     Generated manuscript figures (600 dpi PNG + vector PDF)
habulator-api/               FastAPI inference service (main.py + trained model files)
habulator-web/               Next.js web application
```

---

## Quick Start

### Prerequisites
- Node.js ≥ 18
- Python ≥ 3.10

### Run locally

```bash
git clone https://github.com/rohitqueens-lang/HABulator_Pro.git
cd HABulator_Pro

# Backend (port 8000) — trained model files are included in habulator-api/
cd habulator-api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal, port 3000)
cd habulator-web
npm install
cp .env.example .env.local        # NEXT_PUBLIC_API_URL defaults to http://localhost:8000
npm run dev                        # → http://localhost:3000
```

---

## Inputs & Outputs

### Inputs (6 environmental features)

| Feature | Description | Unit | Valid range |
|---|---|---|---|
| `TEMP` | Surface water temperature | °C | 0 – 30 |
| `TP` | Total phosphorus | µg/L | 0.1 – 200 |
| `SI` | Silica | mg/L | 0 – 5 |
| `NO23` | Nitrate + nitrite | mg/L | 0 – 3 |
| `STN_DEPTH_M` | Station depth | m | 1 – 400 |
| `DOY` | Day of year | — | 1 – 365 |

The API derives three further features before inference — `DOY_sin`, `DOY_cos`
(cyclical season) and `NP_ratio` (= NO23 / TP) — for **8 model features** in total.

### Outputs

| Field | Description |
|---|---|
| `pred_mgL` | Best-estimate biovolume (Duan-corrected) |
| `lower_mgL` | 90% interval lower bound |
| `upper_mgL` | 90% interval upper bound |
| `base_val` | Model baseline (log scale) for SHAP |
| `shap` | Per-feature SHAP values (log scale) |
| `coverage` | Nominal interval coverage (0.90) |

---

## API Reference

Interactive docs at `https://<your-api>.onrender.com/docs`.

```bash
# Health
GET /health

# Predict
POST /predict
Content-Type: application/json

{
  "group": "EDIAT",
  "TEMP": 12.5,
  "TP": 8.3,
  "SI": 1.2,
  "NO23": 0.4,
  "STN_DEPTH_M": 45.0,
  "DOY": 120
}
```

---

## Phytoplankton Groups

| Code | Group | Status |
|---|---|---|
| `EDIAT` | Early Diatoms | ✅ Live |
| `LDIAT` | Late Diatoms | ✅ Live |
| `CHLOR` | Chlorophytes | ✅ Live |
| `CRYPT` | Cryptophytes | ✅ Live |
| `CYANO` | Cyanobacteria | ✅ Live |

---

## Reproducing the model & figures

```bash
pip install -r requirements.txt
python train_emulator.py --target EDIAT      # then LDIAT, CHLOR, CRYPT, CYANO
python make_performance_figure.py            # and the other make_*.py figure scripts
```

Figures are written to `figures/` (600 dpi PNG + vector PDF).

---

## Deployment

### API → Render
1. Push this repo to GitHub.
2. [render.com](https://render.com) → **New Web Service** → connect the repo.
3. **Root directory:** `habulator-api` · **Build:** `pip install -r requirements.txt` ·
   **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT` · **Health check:** `/health`.
   Trained model files are committed, so the service runs immediately.

### Frontend → Vercel
1. [vercel.com](https://vercel.com) → **Add New Project** → import the repo.
2. **Root directory:** `habulator-web`.
3. Environment variable: `NEXT_PUBLIC_API_URL = https://<your-api>.onrender.com`.
4. Name the Vercel project starting with `habulator` — the API's CORS allows `habulator*.vercel.app`.

---

## Methods

- **Model** — XGBoost regressor; Optuna (TPE) hyperparameter optimization with
  station-grouped GroupKFold(5) cross-validation.
- **Uncertainty** — conformalized quantile regression (CQR), 90% nominal coverage;
  empirical held-out coverage 0.91–0.95 across groups.
- **Explainability** — SHAP TreeExplainer (exact Shapley values for tree models),
  on the log(1 + biovolume) model scale.
- **Validation** — predictive skill via station-grouped cross-validation (no station in
  both train and test); coverage via a separate random split (conformal coverage requires
  exchangeability).
- **Data** — U.S. EPA Great Lakes National Program Office (GLNPO / GLENDA), 2001–2021
  (2,316 samples, 73 stations).

---

## Results (held-out)

| Group | Cross-validated R² (log scale) | 90% interval coverage |
|---|:---:|:---:|
| EDIAT | 0.68 | 0.95 |
| CYANO | 0.57 | 0.92 |
| CHLOR | 0.45 | 0.93 |
| CRYPT | 0.35 | 0.91 |
| LDIAT | 0.25 | 0.92 |

---

## Citation

```bibtex
@software{habulator2026,
  author  = {Shukla, Rohit},
  title   = {Habulator: Great Lakes Phytoplankton Biovolume Emulator},
  year    = {2026},
  url     = {https://github.com/rohitqueens-lang/HABulator_Pro},
  note    = {XGBoost emulator with conformalized quantile-regression intervals}
}
```

---

## License

MIT © 2026 Rohit Shukla
