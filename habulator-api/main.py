"""
Habulator FastAPI backend
=========================
Serves phytoplankton biovolume predictions via XGBoost + quantile regression PI.

Model artifacts (LAZY-loaded on first request per group, then cached):
  ediat_model.json              — XGBoost mean booster (point predictions + SHAP)
  ediat_model_lower.json        — XGBoost lower quantile booster (adaptive PI)
  ediat_model_upper.json        — XGBoost upper quantile booster (adaptive PI)
  ediat_webapp_config.json      — smearing_factor, confidence, feature metadata

Cold-start design: startup does NO model loading and runs NO self-test, so the
service becomes healthy in seconds. Each group's artifacts load on its first
/predict (cached thereafter). The golden self-test is available on demand at
GET /selftest (it loads every group and verifies frozen outputs).

Expected JSON keys in ediat_webapp_config.json:
  {
    "smearing_factor": <float>,     // Duan smearing estimate (mean(exp(residuals)))
    "confidence": 0.90,             // PI coverage level
    "alpha": 0.10,                  // miscoverage rate
    "method": "XGBoost quantile regression (adaptive PI)"
  }
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("habulator")

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent

FEATURES = [
    "TEMP",
    "TP",
    "SI",
    "NO23",
    "STN_DEPTH_M",
    "DOY_sin",
    "DOY_cos",
    "NP_ratio",
]

FEAT_LABELS: dict[str, str] = {
    "TEMP": "Temperature",
    "TP": "Total Phosphorus",
    "SI": "Silica",
    "NO23": "Nitrate + Nitrite",
    "STN_DEPTH_M": "Station Depth",
    "DOY_sin": "Seasonality (sin)",
    "DOY_cos": "Seasonality (cos)",
    "NP_ratio": "N:P Ratio",
}

# Input validation bounds (same as frontend)
INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "TEMP": (0.0, 30.0),
    "TP": (0.1, 200.0),
    "SI": (0.0, 5.0),
    "NO23": (0.0, 3.0),
    "STN_DEPTH_M": (1.0, 400.0),
    "DOY": (1.0, 365.0),
}

# ─── Per-group model state ────────────────────────────────────────────────────
GROUPS = ["EDIAT", "LDIAT", "CHLOR", "CRYPT", "CYANO"]


class ModelState:
    def __init__(self) -> None:
        self.booster: xgb.Booster | None = None
        self.booster_lower: xgb.Booster | None = None
        self.booster_upper: xgb.Booster | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.smearing_factor: float = 1.0
        self.confidence: float = 0.90
        self.cqr_correction: float = 0.0  # conformal width Q (log), added to bounds


# Registry: group ID -> ModelState. Populated lazily on first request per group.
_models: dict[str, ModelState] = {}
_load_lock = threading.Lock()  # guards lazy loading against concurrent first requests

# ─── Self-test (on-demand golden-output check; NOT run at startup) ──────────────
# A fixed input whose expected output is frozen in selftest_golden.json. GET /selftest
# recomputes via the SAME predict core and asserts it matches (within tolerance),
# surfacing artifact/code drift. Kept off the startup path to keep cold starts fast.
SELFTEST_INPUT = {"TEMP": 22.0, "TP": 40.0, "SI": 0.3, "NO23": 0.5, "STN_DEPTH_M": 15.0, "DOY": 210}
GOLDEN_PATH = MODEL_DIR / "selftest_golden.json"
SELFTEST_ATOL = 1e-4      # absolute tolerance (mg/L and log units)
SELFTEST_RTOL = 1e-3      # relative tolerance (robust to xgboost patch-version float noise)
_selftest: dict[str, Any] = {"ran": False, "passed": None}


def _model_file_exists(group: str) -> bool:
    return (MODEL_DIR / f"{group.lower()}_model.json").exists()


def _available_groups() -> list[str]:
    """Groups whose artifacts are present on disk (no loading)."""
    return [g for g in GROUPS if _model_file_exists(g)]


def _any_real_artifacts() -> bool:
    return any(_model_file_exists(g) for g in GROUPS)


def _load_group(group: str) -> ModelState | None:
    """Load one group's artifacts. Returns None if the mean model is absent."""
    tag = group.lower()
    model_path = MODEL_DIR / f"{tag}_model.json"
    if not model_path.exists():
        return None

    st = ModelState()
    booster = xgb.Booster(); booster.load_model(str(model_path))
    st.booster = booster

    # Quantile boosters (adaptive PI bounds) — optional
    for fname, attr in [(f"{tag}_model_lower.json", "booster_lower"),
                        (f"{tag}_model_upper.json", "booster_upper")]:
        p = MODEL_DIR / fname
        if p.exists():
            b = xgb.Booster(); b.load_model(str(p)); setattr(st, attr, b)
        else:
            log.warning("%s missing — %s PI will fall back to symmetric", fname, group)

    st.explainer = shap.TreeExplainer(st.booster)

    cfg_path = MODEL_DIR / f"{tag}_webapp_config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg: dict[str, Any] = json.load(f)
        st.smearing_factor = float(cfg.get("smearing_factor", 1.0))
        st.confidence = float(cfg.get("confidence", 0.90))
        st.cqr_correction = float(cfg.get("cqr_correction", 0.0))
    return st


def _get_model(group: str) -> ModelState | None:
    """Return a group's ModelState, lazy-loading + caching on first use.
    Falls back to a synthetic DEMO EDIAT only when no real artifacts exist at all."""
    g = group.upper()
    st = _models.get(g)
    if st is not None:
        return st
    with _load_lock:
        st = _models.get(g)            # double-checked under lock
        if st is not None:
            return st
        if _model_file_exists(g):
            st = _load_group(g)
            if st is not None:
                _models[g] = st
                log.info("Lazy-loaded %s — smearing=%.4f conf=%.2f Q=%.4f",
                         g, st.smearing_factor, st.confidence, st.cqr_correction)
            return st
        # No real artifacts anywhere -> serve a synthetic EDIAT so the app is usable.
        if g == "EDIAT" and not _any_real_artifacts():
            log.warning("No model artifacts found — building synthetic DEMO EDIAT (lazy)")
            demo = ModelState()
            demo.booster = _build_demo_booster()
            demo.explainer = shap.TreeExplainer(demo.booster)
            _models[g] = demo
            return demo
        return None


def _build_demo_booster() -> xgb.Booster:
    """
    Build a minimal XGBoost booster trained on synthetic data for demo/dev mode.
    This is only used when ediat_model.json is absent.
    """
    rng = np.random.default_rng(42)
    n = 500
    X = rng.uniform(
        low=[0, 0.1, 0, 0, 1, -1, -1, 0],
        high=[30, 200, 5, 3, 400, 1, 1, 30],
        size=(n, len(FEATURES)),
    )
    y = (
        0.04 * X[:, 0]          # TEMP
        + 0.008 * X[:, 1]       # TP
        + 0.5
        + rng.normal(0, 0.4, n)
    )
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    params = {
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=80, verbose_eval=False)
    log.info("Demo booster trained on synthetic data (%d samples)", n)
    return booster


# ─── Prediction core (single source of truth: used by /predict AND the self-test) ─
def _predict_core(st: ModelState, TEMP: float, TP: float, SI: float, NO23: float,
                  STN_DEPTH_M: float, DOY: float) -> dict[str, Any]:
    """Run the full prediction for one group. Returns unrounded values + per-feature
    SHAP. The /predict route and the self-test both call this, so they can never
    diverge."""
    doy_rad = 2 * math.pi * DOY / 365.0
    raw_values = {
        "TEMP": TEMP, "TP": TP, "SI": SI, "NO23": NO23, "STN_DEPTH_M": STN_DEPTH_M,
        "DOY_sin": math.sin(doy_rad), "DOY_cos": math.cos(doy_rad),
        "NP_ratio": NO23 / TP if TP > 0 else 0.0,
    }
    X = np.array([[raw_values[f] for f in FEATURES]], dtype=np.float32)
    dmat = xgb.DMatrix(X, feature_names=FEATURES)

    log_hat = float(st.booster.predict(dmat)[0])
    smear = st.smearing_factor
    pred_mgL = max(0.0, smear * math.exp(log_hat) - 1.0)   # Duan smearing on the point

    if st.booster_lower is not None and st.booster_upper is not None:
        q = st.cqr_correction                              # CQR width (log scale)
        log_lo = float(st.booster_lower.predict(dmat)[0]) - q
        log_hi = float(st.booster_upper.predict(dmat)[0]) + q
        lower_mgL = max(0.0, math.exp(log_lo) - 1.0)       # bounds: plain expm1 (no smear)
        upper_mgL = max(0.0, math.exp(log_hi) - 1.0)
    else:
        lower_mgL = max(0.0, pred_mgL * 0.5)
        upper_mgL = pred_mgL * 1.5
    lower_mgL = min(lower_mgL, pred_mgL)
    upper_mgL = max(upper_mgL, pred_mgL)

    shap_matrix = st.explainer.shap_values(X)
    shap_vals = shap_matrix[0] if getattr(shap_matrix, "ndim", 1) == 2 else shap_matrix
    base_val = float(st.explainer.expected_value)
    shap = {f: {"value": float(raw_values[f]), "shap": float(shap_vals[i])}
            for i, f in enumerate(FEATURES)}
    return {"pred_mgL": pred_mgL, "lower_mgL": lower_mgL, "upper_mgL": upper_mgL,
            "base_val": base_val, "shap": shap}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= SELFTEST_ATOL + SELFTEST_RTOL * abs(b)


def _run_selftest() -> dict[str, Any]:
    """Assert a known input reproduces frozen golden outputs for every golden group.
    Lazy-loads each group as needed. Logs PASS/FAIL; result exposed at /selftest and
    cached in /health. Set HABULATOR_STRICT_SELFTEST=1 to raise on mismatch."""
    if not GOLDEN_PATH.exists():
        log.warning("SELF-TEST skipped — %s not found", GOLDEN_PATH.name)
        _selftest.update(ran=False, passed=None, reason="no golden file")
        return _selftest
    golden = json.loads(GOLDEN_PATH.read_text())
    inp, gset = golden["input"], golden["groups"]
    results: dict[str, bool] = {}
    for g in gset:
        st = _get_model(g)
        if st is None or st.booster is None or st.explainer is None:
            continue
        got, exp = _predict_core(st, **inp), gset[g]
        diffs = [k for k in ("pred_mgL", "lower_mgL", "upper_mgL", "base_val")
                 if not _close(got[k], exp[k])]
        diffs += [f"shap[{f}]" for f in FEATURES
                  if not _close(got["shap"][f]["shap"], exp["shap"][f]["shap"])]
        results[g] = not diffs
        if diffs:
            log.error("SELF-TEST FAIL %s — drift in: %s", g, ", ".join(diffs))
        else:
            log.info("SELF-TEST PASS %s  (pred=%.4f, base=%.4f)", g, got["pred_mgL"], got["base_val"])
    passed = bool(results) and all(results.values())
    _selftest.update(ran=True, passed=passed, groups_ok=sum(results.values()),
                     groups_total=len(results), atol=SELFTEST_ATOL, rtol=SELFTEST_RTOL)
    if passed:
        log.info("SELF-TEST PASS — %d/%d groups reproduce golden (atol=%.0e rtol=%.0e)",
                 len(results), len(results), SELFTEST_ATOL, SELFTEST_RTOL)
    else:
        log.error("SELF-TEST FAILED — artifact/code drift detected (%d/%d ok)",
                  sum(results.values()), len(results))
        if os.environ.get("HABULATOR_STRICT_SELFTEST") == "1":
            raise RuntimeError("Self-test failed (strict mode)")
    return _selftest


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # No model loading / self-test here — keep cold starts fast. Models load lazily
    # on first /predict; run the golden check on demand at GET /selftest.
    avail = _available_groups()
    log.info("Habulator API ready — lazy model loading enabled. Available groups: %s",
             ", ".join(avail) or "none (DEMO mode)")
    yield
    log.info("Habulator API shutting down")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Habulator API",
    description="Great Lakes phytoplankton biovolume prediction — XGBoost + Quantile Regression PI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Vercel frontend + local dev
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://habulator.vercel.app",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=r"https://habulator.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    group: str = Field(default="EDIAT", description="Phytoplankton group ID")
    TEMP: float = Field(..., description="Water temperature (°C)")
    TP: float = Field(..., description="Total phosphorus (µg/L)")
    SI: float = Field(..., description="Silica (mg/L)")
    NO23: float = Field(..., description="Nitrate + nitrite (mg/L)")
    STN_DEPTH_M: float = Field(..., description="Station depth (m)")
    DOY: int = Field(..., description="Day of year (1–365)")

    @field_validator("TEMP")
    @classmethod
    def val_temp(cls, v: float) -> float:
        lo, hi = INPUT_BOUNDS["TEMP"]
        if not (lo <= v <= hi):
            raise ValueError(f"TEMP must be between {lo} and {hi} °C")
        return v

    @field_validator("TP")
    @classmethod
    def val_tp(cls, v: float) -> float:
        lo, hi = INPUT_BOUNDS["TP"]
        if not (lo <= v <= hi):
            raise ValueError(f"TP must be between {lo} and {hi} µg/L")
        return v

    @field_validator("SI")
    @classmethod
    def val_si(cls, v: float) -> float:
        lo, hi = INPUT_BOUNDS["SI"]
        if not (lo <= v <= hi):
            raise ValueError(f"SI must be between {lo} and {hi} mg/L")
        return v

    @field_validator("NO23")
    @classmethod
    def val_no23(cls, v: float) -> float:
        lo, hi = INPUT_BOUNDS["NO23"]
        if not (lo <= v <= hi):
            raise ValueError(f"NO23 must be between {lo} and {hi} mg/L")
        return v

    @field_validator("STN_DEPTH_M")
    @classmethod
    def val_depth(cls, v: float) -> float:
        lo, hi = INPUT_BOUNDS["STN_DEPTH_M"]
        if not (lo <= v <= hi):
            raise ValueError(f"STN_DEPTH_M must be between {lo} and {hi} m")
        return v

    @field_validator("DOY")
    @classmethod
    def val_doy(cls, v: int) -> int:
        if not (1 <= v <= 365):
            raise ValueError("DOY must be between 1 and 365")
        return v


class ShapEntry(BaseModel):
    feature: str
    value: float
    shap: float


class PredictResponse(BaseModel):
    pred_mgL: float
    lower_mgL: float
    upper_mgL: float
    base_val: float
    shap: list[ShapEntry]
    coverage: float
    group: str


class HealthResponse(BaseModel):
    status: str
    model: str           # comma-joined available groups (back-compat)
    groups: list[str]    # available group IDs (artifacts present)
    version: str
    demo_mode: bool
    selftest: dict[str, Any]   # on-demand golden-output self-test result (run via /selftest)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    real = _any_real_artifacts()
    avail = _available_groups() if real else ["EDIAT"]   # demo serves a synthetic EDIAT
    # "ok" unless an on-demand self-test has explicitly failed
    healthy = _selftest.get("passed") is not False
    return HealthResponse(
        status="ok" if healthy else "degraded",
        model=",".join(avail) or "none",
        groups=avail,
        version="1.0.0",
        demo_mode=not real,
        selftest=_selftest,
    )


@app.get("/groups", tags=["meta"])
async def groups() -> dict[str, Any]:
    """Available groups with their PI confidence level — read from config without
    loading the boosters (cheap)."""
    out: dict[str, Any] = {}
    for g in _available_groups():
        tag = g.lower()
        conf = 0.90
        cfg_path = MODEL_DIR / f"{tag}_webapp_config.json"
        if cfg_path.exists():
            try:
                conf = float(json.loads(cfg_path.read_text()).get("confidence", 0.90))
            except Exception:
                pass
        adaptive = (MODEL_DIR / f"{tag}_model_lower.json").exists() and \
                   (MODEL_DIR / f"{tag}_model_upper.json").exists()
        out[g] = {"confidence": conf, "adaptive_pi": adaptive}
    return out


@app.get("/selftest", tags=["meta"])
async def selftest() -> dict[str, Any]:
    """Run the golden-output self-test on demand (lazy-loads every group, then
    verifies frozen outputs). Use this to confirm artifact/code integrity."""
    return _run_selftest()


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(req: PredictRequest) -> PredictResponse:
    st = _get_model(req.group.upper())
    if st is None:
        raise HTTPException(
            status_code=404,
            detail=f"Group '{req.group}' not available. Available: {', '.join(_available_groups()) or 'none'}",
        )
    if st.booster is None or st.explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded — try again shortly")

    core = _predict_core(st, req.TEMP, req.TP, req.SI, req.NO23, req.STN_DEPTH_M, req.DOY)

    shap_entries: list[ShapEntry] = [
        ShapEntry(feature=f, value=core["shap"][f]["value"], shap=core["shap"][f]["shap"])
        for f in FEATURES
    ]
    shap_entries.sort(key=lambda e: abs(e.shap), reverse=True)

    return PredictResponse(
        pred_mgL=round(core["pred_mgL"], 6),
        lower_mgL=round(core["lower_mgL"], 6),
        upper_mgL=round(core["upper_mgL"], 6),
        base_val=round(core["base_val"], 6),
        shap=shap_entries,
        coverage=round(st.confidence, 2),
        group=req.group,
    )
