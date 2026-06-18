"""
make_uncertainty_main.py
========================
Uncertainty figure (single 2x3 = 6-panel) for the Habulator XGBoost + CQR pipeline.

  (a) Calibration / reliability — empirical vs nominal coverage for all 5 groups on a
      single held-out split (seed 42); curves should track the diagonal (distribution-
      free coverage guarantee, Romano et al. 2019).
  (b–f) 90% prediction intervals per group — held-out test observations (seed 42) ordered
      by predicted value, with the CQR band (widening with prediction = adaptivity),
      a dashed prediction line, and points coloured covered / missed; coverage annotated.

Single split, seed 42 throughout → fully reproducible. (Coverage is stable across random
splits: 0.90 ± 0.02 at the 90% level over 20 splits — reported as a one-line robustness
statement in the manuscript, not shown here.) Methodology matches train_emulator.py:
quantile bounds on the 70% fit set, conformal Q on the 15% calibration set, expm1 bounds
(no smearing), coverage on the 15% test set. The mean line is the fit-set model, so
nothing scored has seen the test rows. Read-only. Output: uncertainty_main.{pdf,png}
"""
import json, warnings, numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

SEED = 42
ALPHA = 0.10
LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99])
BASE = ["TEMP", "TP", "SI", "NO23", "STN_DEPTH_M", "DOY_sin", "DOY_cos", "NP_ratio"]
GROUPS = ["EDIAT", "CYANO", "CHLOR", "CRYPT", "LDIAT"]
COLOR = {"EDIAT": "#0072B2", "CYANO": "#D55E00", "CHLOR": "#009E73",
         "CRYPT": "#CC79A7", "LDIAT": "#E69F00"}  # Okabe–Ito
BAND = "#0D9488"; PRED = "#08306B"          # band = teal; prediction line = dark navy
COVERED, MISSED = "#3A3F47", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.3,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


def split_seeded(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nf, nc = int(0.70 * n), int(0.15 * n)
    return idx[:nf], idx[nf:nf + nc], idx[nf + nc:]


def conformal_Q(scores, n_cal, level):
    ql = min(np.ceil((n_cal + 1) * level) / n_cal, 1.0)
    return float(np.quantile(scores, ql, method="higher"))


def fit_group(t, d):
    common = dict(json.load(open(f"{t.lower()}_metrics.json"))["best_params"],
                  tree_method="hist", random_state=SEED)
    df = d[d[t].notna()].reset_index(drop=True)
    X, yl = df[BASE], np.log1p(df[t].to_numpy(float))
    fit, cal, te = split_seeded(len(df), SEED)
    mean = xgb.XGBRegressor(objective="reg:squarederror", **common).fit(X.iloc[fit], yl[fit])
    lo = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=ALPHA / 2, **common).fit(X.iloc[fit], yl[fit])
    hi = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=1 - ALPHA / 2, **common).fit(X.iloc[fit], yl[fit])
    scores = np.maximum(lo.predict(X.iloc[cal]) - yl[cal], yl[cal] - hi.predict(X.iloc[cal]))
    return dict(yl_te=yl[te], pred_te=mean.predict(X.iloc[te]),
                qlo_te=lo.predict(X.iloc[te]), qhi_te=hi.predict(X.iloc[te]),
                scores=scores, n_cal=len(cal))


d = pd.read_csv("habulator_master_v2.csv")
d["DOY_sin"] = np.sin(2 * np.pi * d["DOY"] / 365.0)
d["DOY_cos"] = np.cos(2 * np.pi * d["DOY"] / 365.0)
G = {t: fit_group(t, d) for t in GROUPS}

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
axes = axes.ravel()
_mgl = [0.1, 1, 10, 50]

# ── (a) Calibration / reliability (single split, seed 42) ──
axA = axes[0]
axA.plot([0, 1], [0, 1], color="#9AA3AE", lw=0.8, ls="--", zorder=0)   # diagonal = perfect calibration
for t in GROUPS:
    g = G[t]
    emp = [np.mean((g["yl_te"] >= g["qlo_te"] - conformal_Q(g["scores"], g["n_cal"], p)) &
                   (g["yl_te"] <= g["qhi_te"] + conformal_Q(g["scores"], g["n_cal"], p))) for p in LEVELS]
    axA.plot(LEVELS, emp, "-o", color=COLOR[t], lw=1.0, ms=2.6, label=t)
axA.set_xlim(0.45, 1.01); axA.set_ylim(0.45, 1.01)
axA.set_xlabel("Nominal coverage  (1 − α)"); axA.set_ylabel("Empirical coverage (test)")
axA.set_title("a   Calibration", loc="left", fontweight="bold")
axA.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#DDDDDD",
           fontsize=6.0, labelspacing=0.22, handlelength=1.1, borderpad=0.4, handletextpad=0.5)
for sp in ("top", "right"): axA.spines[sp].set_visible(False)

# ── (b–f) per-group 90% prediction intervals (single split, seed 42) ──
letters = "bcdef"
for j, t in enumerate(GROUPS):
    ax = axes[j + 1]; g = G[t]
    Q90 = conformal_Q(g["scores"], g["n_cal"], 1 - ALPHA)
    lo_l = np.maximum(0.0, g["qlo_te"] - Q90); hi_l = g["qhi_te"] + Q90
    covered = (g["yl_te"] >= g["qlo_te"] - Q90) & (g["yl_te"] <= hi_l)
    o = np.argsort(g["pred_te"]); xr = np.arange(len(o))
    ax.fill_between(xr, lo_l[o], hi_l[o], color=BAND, alpha=0.16, lw=0)
    ax.plot(xr, lo_l[o], color=BAND, lw=0.4, alpha=0.5); ax.plot(xr, hi_l[o], color=BAND, lw=0.4, alpha=0.5)
    ax.plot(xr, g["pred_te"][o], color=PRED, lw=0.9, ls="--")             # dashed prediction line
    ax.scatter(xr[covered[o]], g["yl_te"][o][covered[o]], s=4, color=COVERED, alpha=0.6, lw=0)
    ax.scatter(xr[~covered[o]], g["yl_te"][o][~covered[o]], s=6, color=MISSED, lw=0, zorder=4)  # smaller misses
    ax.set_yticks([np.log1p(v) for v in _mgl]); ax.set_yticklabels([f"{v:g}" for v in _mgl])
    ax.set_ylim(-0.15, 4.5); ax.set_xlim(-5, len(xr) + 5)
    ax.set_title(f"{letters[j]}   {t}", loc="left", fontweight="bold")
    ax.text(0.04, 0.95, f"coverage = {covered.mean():.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.0, color="#4C5663")
    if (j + 1) in (1, 3):                      # left-column ribbons (pos1=b, pos3=d)
        ax.set_ylabel("Biovolume (mg/L)")
    if (j + 1) in (3, 4, 5):                   # bottom row
        ax.set_xlabel("Test observations (ordered by prediction)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# ── shared interval legend (figure-level, bottom) ──
handles = [Patch(facecolor=BAND, alpha=0.3, edgecolor="none", label="90% interval"),
           Line2D([0], [0], color=PRED, lw=1.1, ls="--", label="prediction"),
           Line2D([0], [0], marker="o", ls="none", markerfacecolor=COVERED, markeredgecolor="none", markersize=4, label="covered"),
           Line2D([0], [0], marker="o", ls="none", markerfacecolor=MISSED, markeredgecolor="none", markersize=5, label="missed")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.005), fontsize=7, handletextpad=0.5, columnspacing=1.8)

fig.tight_layout(rect=[0, 0.04, 1, 1.0])
fig.savefig("figures/uncertainty_main.pdf", bbox_inches="tight")
fig.savefig("figures/uncertainty_main.png", bbox_inches="tight")

print("Per-group 90% coverage (single split, seed 42):")
for t in GROUPS:
    g = G[t]; Q90 = conformal_Q(g["scores"], g["n_cal"], 1 - ALPHA)
    cov = np.mean((g["yl_te"] >= g["qlo_te"] - Q90) & (g["yl_te"] <= g["qhi_te"] + Q90))
    print(f"  {t:6s} coverage={cov:.3f}")
print("Saved: uncertainty_main.{pdf,png}")
