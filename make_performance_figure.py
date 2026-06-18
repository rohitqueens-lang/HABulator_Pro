"""
make_performance_figure.py
==========================
PNAS/Science-quality XGBoost performance figure for all 5 phytoplankton groups.

Honest, cross-validated: predictions are GroupKFold(5) OUT-OF-FOLD (by station) with
each group's tuned best_params — i.e., predictions for held-out stations, matching the
headline cross-validated skill (no in-sample inflation). Plotted on the model
(log1p) scale, the scale on which skill is reported.

Convention (Piñeiro et al., 2008): observed vs predicted with the 1:1 identity line
(no fitted regression line, which would mislead the slope interpretation).

Panels a–e: observed vs predicted per group (scatter), with Pearson r, RMSE, and MAE —
all computed on the log1p scale. 6th cell: 1:1 / OLS-fit line key.
Output: performance_obs_pred.{pdf,png}
"""
import json, warnings, numpy as np, pandas as pd, xgboost as xgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

SEED = 42
BASE = ["TEMP", "TP", "SI", "NO23", "STN_DEPTH_M", "DOY_sin", "DOY_cos", "NP_ratio"]
GROUPS = ["EDIAT", "LDIAT", "CHLOR", "CRYPT", "CYANO"]
TAXON = {"EDIAT": "Early diatoms", "LDIAT": "Late diatoms",
         "CHLOR": "Chlorophytes", "CRYPT": "Cryptophytes", "CYANO": "Cyanobacteria"}
COLOR = {"EDIAT": "#0072B2", "CYANO": "#D55E00", "CHLOR": "#009E73",
         "CRYPT": "#CC79A7", "LDIAT": "#E69F00"}  # Okabe–Ito (colorblind-safe)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

FIT_COLOR = "#2563EB"   # OLS fit line (dashed)

# Axes use log1p geometry but are tick-labeled in absolute mg/L (interpretable units,
# handles the orders-of-magnitude range and zeros). Metrics stay on the log scale.
MGL_TICKS = [0.1, 1, 10, 50, 100]
def set_mgl_axes(ax, lim):
    pos = [np.log1p(c) for c in MGL_TICKS if lim[0] <= np.log1p(c) <= lim[1]]
    labs = [f"{c:g}" for c in MGL_TICKS if lim[0] <= np.log1p(c) <= lim[1]]
    ax.set_xticks(pos); ax.set_xticklabels(labs)
    ax.set_yticks(pos); ax.set_yticklabels(labs)

d = pd.read_csv("habulator_master_v2.csv")
d["DOY_sin"] = np.sin(2 * np.pi * d["DOY"] / 365.0)
d["DOY_cos"] = np.cos(2 * np.pi * d["DOY"] / 365.0)


def oof(group):
    bp = dict(json.load(open(f"{group.lower()}_metrics.json"))["best_params"],
              objective="reg:squarederror", tree_method="hist", random_state=SEED)
    df = d[d[group].notna()].reset_index(drop=True)
    X, yl, g = df[BASE], np.log1p(df[group].to_numpy(float)), df["station_norm"]
    pred = np.full(len(yl), np.nan)
    for tr, va in GroupKFold(5).split(X, yl, groups=g):
        pred[va] = xgb.XGBRegressor(**bp).fit(X.iloc[tr], yl[tr]).predict(X.iloc[va])
    return {"ylog": yl, "plog": pred}


RES = {g: oof(g) for g in GROUPS}
order = sorted(GROUPS, key=lambda g: pearsonr(RES[g]["ylog"], RES[g]["plog"]).statistic, reverse=True)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
axes = axes.ravel()
letters = "abcde"
for k, g in enumerate(order):
    ax = axes[k]; yobs, ypred = RES[g]["ylog"], RES[g]["plog"]
    r = float(pearsonr(yobs, ypred).statistic)                 # Pearson correlation (log scale)
    rmse = float(np.sqrt(mean_squared_error(yobs, ypred)))     # root mean squared error (log scale)
    mae = float(mean_absolute_error(yobs, ypred))              # mean absolute error (log scale)
    ax.scatter(yobs, ypred, color=COLOR[g], s=6, alpha=0.35, lw=0, rasterized=True)
    lim = [min(yobs.min(), ypred.min()), max(yobs.max(), ypred.max())]
    pad = 0.04 * (lim[1] - lim[0]); lim = [lim[0] - pad, lim[1] + pad]
    xs = np.array(lim)
    ax.plot(xs, xs, color="#1A1A1A", lw=1.0, ls="-", zorder=2)                   # solid 1:1
    m_, b_ = np.polyfit(yobs, ypred, 1)                                          # OLS trend fit
    ax.plot(xs, m_ * xs + b_, color=FIT_COLOR, lw=1.1, ls="--", zorder=3)  # dashed fit
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", "box")
    set_mgl_axes(ax, lim)
    ax.set_title(f"{letters[k]}  {g}", loc="left", fontweight="bold", fontsize=8.3)
    ax.text(0.04, 0.96,
            f"r = {r:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.5, color="#222",
            linespacing=1.35)
    if k % 3 == 0:
        ax.set_ylabel("Predicted biovolume (mg/L)")
    if k >= 2:
        ax.set_xlabel("Observed biovolume (mg/L)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# 6th cell — key: 1:1 / OLS-fit line legend (figure has 5 data panels)
axf = axes[5]; axf.axis("off")
axf.legend([Line2D([0], [0], color="#1A1A1A", lw=1.0, ls="-"),
            Line2D([0], [0], color=FIT_COLOR, lw=1.1, ls="--")],
           ["1:1 line", "OLS fit"], loc="center",
           frameon=False, fontsize=8, handlelength=1.9)

fig.suptitle("XGBoost emulator performance — observed vs cross-validated prediction",
             fontsize=9.3, fontweight="bold", y=1.005)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("figures/performance_obs_pred.pdf", bbox_inches="tight")
fig.savefig("figures/performance_obs_pred.png", bbox_inches="tight")

print("Per-group CV (out-of-fold, log scale):")
for g in order:
    yo, yp_ = RES[g]["ylog"], RES[g]["plog"]
    print(f"  {g:6s} r={pearsonr(yo,yp_).statistic:.3f}  RMSE={np.sqrt(mean_squared_error(yo,yp_)):.3f}  "
          f"MAE={mean_absolute_error(yo,yp_):.3f}  n={len(yo)}")
print("Saved: performance_obs_pred.{pdf,png}")
