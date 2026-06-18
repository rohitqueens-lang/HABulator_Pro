"""
train_emulator.py
=================
Canonical training pipeline for the Habulator phytoplankton **predictive
platform** (environment -> phytogroup biovolume response surface).
Produces the deployable per-group artifacts (models, configs, metrics).

Methodology (paper-standard; predictive-platform framing, NOT temporal forecasting)
───────────────────────────────────────────────────────────────────────────────
TWO-TRACK validation — the skill claim and the coverage claim need different,
each-valid designs:

  TRACK A — SKILL (generalization): group-aware split BY MONITORING STATION so no
    station appears in both train and test. Controls spatial pseudoreplication,
    which a plain random split would let inflate apparent skill. Reported numbers:
    R²/RMSE(log) on group-held-out stations + GroupKFold(5) CV. This is the honest,
    conservative generalization claim.

  TRACK B — INTERVAL COVERAGE: split-conformal (CQR, Romano et al. 2019) requires
    calibration↔test EXCHANGEABILITY, which station-grouping violates (verified:
    grouped calibration undercovers). So the intervals are calibrated on a RANDOM
    (row-level) split — the valid setting for a "new observation from the monitored
    system" marginal-coverage guarantee. The quantile regressors are fit on the 70%
    fit set, CQR-calibrated on 15%, and coverage is reported on the 15% test set.

  DEPLOYED POINT MODEL — refit on ALL rows. Generalization skill is established by
    Track A (GroupKFold-by-station CV); the shipped point model is therefore refit on
    the full dataset for the best estimate. The CQR interval is point-model-INDEPENDENT
    (it depends only on the quantile regressors + Q), so refitting the mean leaves the
    coverage guarantee intact; the API clamps the interval to contain the point estimate,
    which can only widen it (coverage stays ≥ the reported value).

  * Target log1p-transformed; XGBoost handles NaN features natively (no imputation).
  * Optuna TPE HPO with GroupKFold(5) by station selects params (used in both tracks).
  * Point estimate : mean booster (refit on ALL rows) + Duan smearing back-transform
        E[y|x] = smear * exp(f(x)) - 1,  smear = mean(exp(residuals_log1p)).

Exports (into habulator-api/), consumed by main.py:
    {tag}_model.json          mean booster — refit on ALL rows (point + SHAP)
    {tag}_model_lower.json    lower quantile booster
    {tag}_model_upper.json    upper quantile booster
    {tag}_webapp_config.json  {smearing_factor, confidence, method, cqr_correction,
                               features, feature_limits, feature_labels, target}
Also writes {tag}_metrics.json (root) with the numbers for the paper.

Usage:
    python3 train_emulator.py --target EDIAT --trials 50
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
MASTER       = "habulator_master_v2.csv"  # canonical (v3 per-group-max dedup tested, score-neutral under HPO)
API_DIR      = Path("habulator-api")
GROUP_COL    = "station_norm"
RANDOM_SEED  = 42

FEATURES = ["TEMP", "TP", "SI", "NO23", "STN_DEPTH_M", "DOY_sin", "DOY_cos", "NP_ratio"]

FEATURE_LIMITS = {
    "TEMP":        (0.0,  30.0),
    "TP":          (0.1, 200.0),
    "SI":          (0.0,   5.0),
    "NO23":        (0.0,   3.0),
    "STN_DEPTH_M": (1.0, 400.0),
}
FEATURE_LABELS = {
    "TEMP": "Water Temperature (°C)",
    "TP": "Total Phosphorus (µg/L)",
    "SI": "Silica (mg/L)",
    "NO23": "Nitrate+Nitrite (mg/L)",
    "STN_DEPTH_M": "Station Depth (m)",
    "DOY_sin": "Day-of-Year (sin)",
    "DOY_cos": "Day-of-Year (cos)",
    "NP_ratio": "N:P Ratio",
}


def load_data(target: str) -> pd.DataFrame:
    df = pd.read_csv(MASTER)
    df["DOY_sin"] = np.sin(2 * np.pi * df["DOY"] / 365.0)
    df["DOY_cos"] = np.cos(2 * np.pi * df["DOY"] / 365.0)
    df = df[df[target].notna()].reset_index(drop=True)  # keep NaN features (XGB handles)
    return df


def group_split(df: pd.DataFrame):
    """TRACK A — 85/15 train/test by station group (seed-stable). For skill."""
    groups = df[GROUP_COL]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return train_idx, test_idx


def random_split(n: int):
    """TRACK B — 70/15/15 fit/calib/test random row split (exchangeable). For CQR."""
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(n)
    n_fit, n_cal = int(0.70 * n), int(0.15 * n)
    return idx[:n_fit], idx[n_fit:n_fit + n_cal], idx[n_fit + n_cal:]


def run_hpo(X, y_log, groups, n_trials: int):
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y_log, groups=groups))

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "gamma":            trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
        }
        rmses = []
        for tr, va in folds:
            m = xgb.XGBRegressor(**params)
            m.fit(X.iloc[tr], y_log[tr])
            pred = m.predict(X.iloc[va])
            rmses.append(np.sqrt(mean_squared_error(y_log[va], pred)))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="EDIAT")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--confidence", type=float, default=0.90)
    args = ap.parse_args()

    target = args.target.upper()
    alpha = 1.0 - args.confidence
    tag = target.lower()

    print("=" * 70)
    print(f"Habulator emulator — target={target}  confidence={args.confidence}  trials={args.trials}")
    print("=" * 70)

    df = load_data(target)
    X = df[FEATURES]
    y = df[target].to_numpy(dtype=float)
    y_log = np.log1p(y)

    n_stat = lambda idx: df.iloc[idx][GROUP_COL].nunique()

    # ── HPO (group CV — params used by both tracks) ──
    g_tr, g_te = group_split(df)
    assert not (set(df.iloc[g_tr][GROUP_COL]) & set(df.iloc[g_te][GROUP_COL]))
    print(f"\nRunning Optuna HPO ({args.trials} trials × 5-fold GroupKFold)...")
    best_params, cv_rmse = run_hpo(
        X.iloc[g_tr], y_log[g_tr], df.iloc[g_tr][GROUP_COL].to_numpy(), args.trials
    )
    print(f"  best group-CV RMSE (log) : {cv_rmse:.4f}")
    common = dict(best_params)
    common.update(tree_method="hist", random_state=RANDOM_SEED)

    def fit_mean(idx):
        m = xgb.XGBRegressor(objective="reg:squarederror", **common)
        m.fit(X.iloc[idx], y_log[idx])
        return m

    # ══ TRACK A — SKILL (group split by station) ══════════════════════════════
    print(f"\n[Track A — skill]  group split: train {len(g_tr)} ({n_stat(g_tr)} stn)"
          f" / test {len(g_te)} ({n_stat(g_te)} stn)")
    skill_model = fit_mean(g_tr)
    smear_g = float(np.mean(np.exp(y_log[g_tr] - skill_model.predict(X.iloc[g_tr]))))
    p_log = skill_model.predict(X.iloc[g_te])
    p_raw = np.maximum(0.0, smear_g * np.exp(p_log) - 1.0)
    r2_log = float(r2_score(y_log[g_te], p_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_log[g_te], p_log)))
    r2_raw = float(r2_score(y[g_te], p_raw))
    rmse_raw = float(np.sqrt(mean_squared_error(y[g_te], p_raw)))
    print(f"  single-split: R²(log)={r2_log:.3f}  RMSE(log)={rmse_log:.3f}  R²(raw)={r2_raw:.3f}")

    # Headline skill = GroupKFold(5) CV (more stable than one 11-station split).
    # Out-of-fold predictions over ALL data with tuned params (params not refit per fold).
    oof = np.full(len(y_log), np.nan)
    for tr_i, va_i in GroupKFold(n_splits=5).split(X, y_log, groups=df[GROUP_COL]):
        fm = xgb.XGBRegressor(objective="reg:squarederror", **common)
        fm.fit(X.iloc[tr_i], y_log[tr_i])
        oof[va_i] = fm.predict(X.iloc[va_i])
    cv_r2_log = float(r2_score(y_log, oof))
    cv_rmse_log_oof = float(np.sqrt(mean_squared_error(y_log, oof)))
    # Overfit diagnostic: in-sample train R² vs CV.
    full_model = xgb.XGBRegressor(objective="reg:squarederror", **common).fit(X, y_log)
    train_r2_log = float(r2_score(y_log, full_model.predict(X)))
    overfit_gap = train_r2_log - cv_r2_log
    gflag = "OK" if overfit_gap <= 0.20 else ("moderate" if overfit_gap <= 0.30 else "LARGE")
    print(f"  GroupKFold CV (headline): R²(log)={cv_r2_log:.3f}  RMSE(log)={cv_rmse_log_oof:.3f}")
    print(f"  Overfit gap: train R²={train_r2_log:.3f}  CV R²={cv_r2_log:.3f}  gap={overfit_gap:.3f} [{gflag}]")

    # ══ TRACK B — INTERVAL COVERAGE (random exchangeable split) ════════════════
    r_fit, r_cal, r_te = random_split(len(df))
    print(f"\n[Track B — coverage]  random split: fit {len(r_fit)} / cal {len(r_cal)} / test {len(r_te)}")
    # Quantile regressors for CQR — fit on r_fit, kept DISJOINT from the calibration set
    # (refitting these on all data would leak r_cal and void the coverage guarantee).
    lo_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha / 2, **common)
    hi_model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=1 - alpha / 2, **common)
    lo_model.fit(X.iloc[r_fit], y_log[r_fit])
    hi_model.fit(X.iloc[r_fit], y_log[r_fit])

    # CQR conformalization on the (exchangeable) random calibration set
    scores = np.maximum(lo_model.predict(X.iloc[r_cal]) - y_log[r_cal],
                        y_log[r_cal] - hi_model.predict(X.iloc[r_cal]))
    n_cal = len(scores)
    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    Q = float(np.quantile(scores, q_level, method="higher"))

    def back_bound(v):   # quantile bounds back-transform with plain expm1 (NO smearing)
        return np.maximum(0.0, np.expm1(v))

    # Coverage of the conformal interval [q_lo - Q, q_hi + Q] on the random test set.
    # This is the Romano et al. (2019) marginal-coverage guarantee — POINT-MODEL-INDEPENDENT,
    # so refitting the mean below cannot affect it. (The API additionally clamps the interval
    # to contain the point estimate, which can only widen it → deployed coverage ≥ this.)
    lo_te = back_bound(lo_model.predict(X.iloc[r_te]) - Q)
    hi_te = back_bound(hi_model.predict(X.iloc[r_te]) + Q)
    coverage = float(np.mean((y[r_te] >= lo_te) & (y[r_te] <= hi_te)))
    mean_width = float(np.mean(hi_te - lo_te))
    print(f"  CQR Q(log)={Q:.4f}  conformal coverage={coverage:.3f} (target {args.confidence})"
          f"  mean width={mean_width:.4f} mg/L")

    # ══ DEPLOYED POINT MODEL — refit on ALL rows after validation ══════════════
    # Generalization skill is established by Track A (GroupKFold-by-station CV); we therefore
    # ship the point model refit on the FULL dataset for the best estimate. The conformal
    # interval above is point-model-independent, so its coverage guarantee is preserved.
    deploy_mean = xgb.XGBRegressor(objective="reg:squarederror", **common).fit(X, y_log)
    smear = float(np.mean(np.exp(y_log - deploy_mean.predict(X))))   # Duan smearing, full-data residuals
    print(f"  Deployed point model: refit on all {len(df)} rows  smear={smear:.4f}")

    # ── Export deployed artifacts (point=all-rows; intervals=split-CQR) ──
    API_DIR.mkdir(exist_ok=True)
    deploy_mean.get_booster().save_model(str(API_DIR / f"{tag}_model.json"))
    lo_model.get_booster().save_model(str(API_DIR / f"{tag}_model_lower.json"))
    hi_model.get_booster().save_model(str(API_DIR / f"{tag}_model_upper.json"))

    config = {
        "smearing_factor": smear, "confidence": args.confidence,
        "method": "CQR (random-split calibration); point model refit on full data", "cqr_correction": Q,
        "features": FEATURES,
        "feature_limits": {k: list(v) for k, v in FEATURE_LIMITS.items()},
        "feature_labels": FEATURE_LABELS, "target": target,
    }
    with open(API_DIR / f"{tag}_webapp_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "target": target,
        "skill_track": {
            "design": "group-by-station; headline = GroupKFold(5) CV (leakage-controlled)",
            "cv_r2_log": cv_r2_log, "cv_rmse_log": cv_rmse_log_oof,  # HEADLINE
            "train_r2_log": train_r2_log, "overfit_gap_log": overfit_gap,
            "single_split_r2_log": r2_log, "single_split_rmse_log": rmse_log,
            "single_split_r2_raw": r2_raw, "single_split_rmse_raw": rmse_raw,
            "hpo_cv_rmse_log": cv_rmse,
            "n_test_split": int(len(g_te)), "stations_test_split": int(n_stat(g_te)),
        },
        "coverage_track": {
            "design": "intervals: quantile regressors on random 70% fit, CQR-calibrated on "
                      "15%, conformal coverage on 15% test (point-model-independent)",
            "n_fit": int(len(r_fit)), "n_calib": int(len(r_cal)), "n_test": int(len(r_te)),
            "confidence": args.confidence, "cqr_correction": Q,
            "coverage_test": coverage, "mean_width_test": mean_width,
            "deployed_point_model": f"refit on all {len(df)} rows (skill from Track-A CV)",
            "deployed_point_model_n": int(len(df)),
        },
        "best_params": best_params,
        "smearing_factor": smear, "seed": RANDOM_SEED,
    }
    with open(f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nExported deployed artifacts -> {API_DIR}/  and  {tag}_metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
