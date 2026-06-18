"""
make_shap_figures.py
====================
PNAS/Science-quality SHAP figures for ALL FIVE phytoplankton groups, using the
**canonical XGBoost pipeline** (NOT ExtraTrees).

Correctness / no-confusion guarantees
-------------------------------------
  * MODEL = XGBoost with each group's tuned best_params (from {group}_metrics.json,
    the exact hyperparameters of the deployed models), refit on the FULL
    habulator_master_v2.csv with log1p target — the standard "final model" for
    interpretation. objective=reg:squarederror, tree_method=hist, seed=42.
  * DATA = habulator_master_v2.csv, the canonical 8 features (NO Secchi — that was
    the old v1 pipeline). Feature order matches the deployed booster exactly.
  * SHAP = exact TreeExplainer. Values are on the model output scale = log1p(biovolume).
    Explaining model behaviour over all observations (standard for SHAP summaries).
Outputs (600 dpi PNG + vector PDF):
  shap_xgb_beeswarm_all.{pdf,png}     5-panel beeswarm (main figure)
  shap_xgb_importance_heatmap.{pdf,png}  cross-group mean|SHAP| heatmap
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import xgboost as xgb, shap
warnings.filterwarnings("ignore")

MASTER   = "habulator_master_v2.csv"
TARGETS  = ["EDIAT", "LDIAT", "CHLOR", "CRYPT", "CYANO"]
FEATURES = ["TEMP", "TP", "SI", "NO23", "STN_DEPTH_M", "DOY_sin", "DOY_cos", "NP_ratio"]
FEAT_LABEL = {
    "TEMP": "Temperature", "TP": "Total phosphorus", "SI": "Silica",
    "NO23": "Nitrate+nitrite", "STN_DEPTH_M": "Station depth",
    "DOY_sin": "Season (sin)", "DOY_cos": "Season (cos)", "NP_ratio": "N:P ratio",
}
TAXON = {"EDIAT": "Early diatoms", "LDIAT": "Late diatoms",
         "CHLOR": "Chlorophytes", "CRYPT": "Cryptophytes", "CYANO": "Cyanobacteria"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})
CMAP = plt.get_cmap("viridis")   # sequential, perceptually-uniform, colorblind-safe
BARCOLOR = "azure"               # importance bars (azure fill + light grey border)
BARSTYLE = "fill"                # "fill" (filled bar) or "line" (lollipop line)


def fit_and_explain(t, d):
    df = d[d[t].notna()].reset_index(drop=True)
    X = df[FEATURES]
    y = np.log1p(df[t].to_numpy(float))
    bp = json.load(open(f"{t.lower()}_metrics.json"))["best_params"]
    model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             random_state=42, **bp).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X)
    return X.to_numpy(float), sv


def beeswarm_offsets(x, half_width=0.42, nbins=60):
    """Vertical offsets for a density-aware beeswarm (dense bins spread wider)."""
    y = np.zeros(len(x))
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return y
    edges = np.linspace(np.nanmin(x[finite]), np.nanmax(x[finite]), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx[finite], minlength=nbins)
    maxc = max(counts.max(), 1)
    rng = np.random.default_rng(0)
    for b in range(nbins):
        m = np.where(idx == b)[0]
        if len(m) <= 1:
            continue
        w = half_width * (counts[b] / maxc)
        offs = np.linspace(-w, w, len(m)); rng.shuffle(offs)
        y[m] = offs
    return y


def norm_color(v):
    """Per-feature 5-95 percentile normalisation for color (robust to outliers)."""
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return np.full_like(v, 0.5, dtype=float)
    lo, hi = np.nanpercentile(v[finite], [5, 95])
    if hi <= lo:
        hi = lo + 1e-9
    return np.clip((v - lo) / (hi - lo), 0, 1)


def panel(ax, X, sv, letter, title):
    imp = np.abs(sv).mean(0)                        # mean|SHAP| per feature (global importance)
    order = np.argsort(imp)[::-1]                   # most important on top
    bartrans = blended_transform_factory(ax.transAxes, ax.transData)  # x: axes-frac, y: data
    immax = imp.max() if imp.max() > 0 else 1.0
    for row, fi in enumerate(order):
        ypos = len(order) - 1 - row                # top = most important
        # background relative-importance bar, anchored at the left axis (variable names)
        frac = imp[fi] / immax * 0.97
        if BARSTYLE == "line":
            ax.plot([0, frac], [ypos, ypos], transform=bartrans, color=BARCOLOR, lw=2.2,
                    alpha=0.95, zorder=0, solid_capstyle="round")
            ax.plot([frac], [ypos], transform=bartrans, marker="o", color=BARCOLOR, ms=3.2,
                    alpha=0.95, zorder=0)
        else:
            ax.add_patch(Rectangle((0, ypos - 0.37), frac, 0.74, transform=bartrans,
                                   facecolor=BARCOLOR, edgecolor="#d2d2d2", linewidth=0.5,
                                   alpha=0.9, zorder=0))   # filled bar with a soft grey border
        s = sv[:, fi]
        off = beeswarm_offsets(s)
        c = norm_color(X[:, fi])
        nan = ~np.isfinite(X[:, fi])
        ax.scatter(s[~nan], ypos + off[~nan], c=c[~nan], cmap=CMAP, vmin=0, vmax=1,
                   s=4, alpha=0.75, linewidths=0, rasterized=True)
        if nan.any():  # missing feature values shown grey (distinct from viridis teal mid-range)
            ax.scatter(s[nan], ypos + off[nan], color="#bdbdbd", s=4, alpha=0.6,
                       linewidths=0, rasterized=True)
    ax.axvline(0, color="#444444", lw=0.6, zorder=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FEAT_LABEL[FEATURES[fi]] for fi in order[::-1]])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_title(title, fontweight="bold", pad=3)
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    d = pd.read_csv(MASTER)
    d["DOY_sin"] = np.sin(2 * np.pi * d["DOY"] / 365.0)
    d["DOY_cos"] = np.cos(2 * np.pi * d["DOY"] / 365.0)

    data = {t: fit_and_explain(t, d) for t in TARGETS}

    # ── Figure 1: 5-panel beeswarm ──
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    axes = axes.ravel()
    letters = "abcde"
    for k, t in enumerate(TARGETS):
        X, sv = data[t]
        panel(axes[k], X, sv, letters[k], t)   # group code title (consistent with other figures)
        if k >= 2:
            axes[k].set_xlabel("SHAP value (log scale)")
    # 6th cell: colourbar (top) + keys — swatches centred at x=0.27, labels at x=0.36
    cax_host = axes[5]; cax_host.axis("off")
    cbar_ax = cax_host.inset_axes([0.12, 0.72, 0.76, 0.05])     # colourbar moved up
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                      cax=cbar_ax, orientation="horizontal")
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", fontsize=7, labelpad=1)        # label tight to the bar
    # relative-importance key (matches the bar style)
    if BARSTYLE == "line":
        cax_host.plot([0.21, 0.31], [0.42, 0.42], color=BARCOLOR, lw=2.2, solid_capstyle="round",
                      transform=cax_host.transAxes, clip_on=False)
        cax_host.plot([0.31], [0.42], marker="o", color=BARCOLOR, ms=3.2,
                      transform=cax_host.transAxes, clip_on=False)
    else:
        cax_host.add_patch(Rectangle((0.21, 0.39), 0.12, 0.06, transform=cax_host.transAxes,
                                     facecolor=BARCOLOR, edgecolor="#d2d2d2", linewidth=0.5,
                                     alpha=0.9, clip_on=False))
    cax_host.text(0.36, 0.42, "relative importance", transform=cax_host.transAxes,
                  ha="left", va="center", fontsize=6.6, color="#555555")
    cax_host.scatter([0.27], [0.30], s=16, color="#bdbdbd", transform=cax_host.transAxes, clip_on=False)
    cax_host.text(0.36, 0.30, "missing feature value", transform=cax_host.transAxes,
                  ha="left", va="center", fontsize=6.6, color="#555555")
    fig.suptitle("SHAP feature attributions per phytoplankton group (XGBoost)",
                 fontsize=9.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("figures/shap_xgb_beeswarm_all.pdf", bbox_inches="tight")
    fig.savefig("figures/shap_xgb_beeswarm_all.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: cross-group mean|SHAP| heatmap (normalised per group) ──
    M = np.array([[np.abs(data[t][1][:, j]).mean() for t in TARGETS]
                  for j in range(len(FEATURES))])   # features × groups
    Mn = M / M.max(axis=0, keepdims=True)           # per-group relative importance
    fig2, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(Mn, cmap=CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels(TARGETS, rotation=0)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEAT_LABEL[f] for f in FEATURES])
    for i in range(len(FEATURES)):
        for j in range(len(TARGETS)):
            ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if Mn[i, j] < 0.6 else "black")
    cb2 = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb2.set_label("Relative mean |SHAP| (per group)", fontsize=7)
    ax.set_title("Cross-group feature importance (XGBoost)", fontweight="bold", fontsize=9)
    fig2.tight_layout()
    fig2.savefig("figures/shap_xgb_importance_heatmap.pdf", bbox_inches="tight")
    fig2.savefig("figures/shap_xgb_importance_heatmap.png", bbox_inches="tight")
    plt.close(fig2)

    # ── text summary (top-3 drivers per group) ──
    print("Top SHAP drivers per group (mean|SHAP|, log scale):")
    for t in TARGETS:
        X, sv = data[t]
        imp = np.abs(sv).mean(0)
        order = np.argsort(imp)[::-1][:3]
        drivers = ", ".join(f"{FEATURES[i]} ({imp[i]:.3f})" for i in order)
        print(f"  {t:6s}: {drivers}")
    print("\nSaved: shap_xgb_beeswarm_all.{pdf,png}  shap_xgb_importance_heatmap.{pdf,png}")


if __name__ == "__main__":
    main()
