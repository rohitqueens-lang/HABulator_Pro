"""
make_shap_dependence.py
=======================
SUPPLEMENTARY figure: top-4 SHAP dependence plots per phytoplankton group (XGBoost).

For each group, the 4 most important features (by mean|SHAP|) are shown as simple
single-colour dependence scatters — SHAP value (log1p model scale) vs feature value —
in the group's colour. No interaction colouring: this is the marginal response shape of
each group's top drivers, complementing the main beeswarm (make_shap_figures.py).

Same model/data as make_shap_figures.py: per-group XGBoost (tuned best_params from
{group}_metrics.json, log1p target, seed 42), exact TreeExplainer SHAP, full
habulator_master_v2.csv. Output: shap_xgb_dependence_top4.{pdf,png}
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb, shap
warnings.filterwarnings("ignore")

MASTER = "habulator_master_v2.csv"
FEATURES = ["TEMP", "TP", "SI", "NO23", "STN_DEPTH_M", "DOY_sin", "DOY_cos", "NP_ratio"]
FEAT_LABEL = {
    "TEMP": "Temperature", "TP": "Total phosphorus", "SI": "Silica",
    "NO23": "Nitrate+nitrite", "STN_DEPTH_M": "Station depth",
    "DOY_sin": "Season (sin)", "DOY_cos": "Season (cos)", "NP_ratio": "N:P ratio",
}
TARGETS = ["EDIAT", "LDIAT", "CHLOR", "CRYPT", "CYANO"]   # row order matches the beeswarm
COLOR = {"EDIAT": "#0072B2", "LDIAT": "#E69F00", "CHLOR": "#009E73",
         "CRYPT": "#CC79A7", "CYANO": "#D55E00"}          # Okabe–Ito (consistent across figures)
TOPK = 4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})


def fit_and_explain(t, d):
    df = d[d[t].notna()].reset_index(drop=True)
    X = df[FEATURES]
    y = np.log1p(df[t].to_numpy(float))
    bp = json.load(open(f"{t.lower()}_metrics.json"))["best_params"]
    model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             random_state=42, **bp).fit(X, y)
    sv = shap.TreeExplainer(model).shap_values(X.to_numpy(float))
    return X.to_numpy(float), sv


def main():
    d = pd.read_csv(MASTER)
    d["DOY_sin"] = np.sin(2 * np.pi * d["DOY"] / 365.0)
    d["DOY_cos"] = np.cos(2 * np.pi * d["DOY"] / 365.0)

    fig, axes = plt.subplots(len(TARGETS), TOPK, figsize=(8.8, 9.4))
    for r, t in enumerate(TARGETS):
        Xa, sv = fit_and_explain(t, d)
        order = np.argsort(np.abs(sv).mean(0))[::-1][:TOPK]
        for c, j in enumerate(order):
            ax = axes[r, c]; xj = Xa[:, j]; sj = sv[:, j]
            ok = np.isfinite(xj); xj, sj = xj[ok], sj[ok]
            lo, hi = np.percentile(xj, [1, 99]); mm = (xj >= lo) & (xj <= hi)   # trim sparse tails
            ax.scatter(xj[mm], sj[mm], color=COLOR[t], s=5, alpha=0.35, lw=0, rasterized=True)
            ax.axhline(0, color="#999999", lw=0.5, zorder=0)
            ax.set_xlabel(FEAT_LABEL[FEATURES[j]], fontsize=7)
            if c == 0:
                ax.set_ylabel(t, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Top-4 SHAP dependence per phytoplankton group   (y-axis: SHAP value, log scale)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig("figures/shap_xgb_dependence_top4.pdf", bbox_inches="tight")
    fig.savefig("figures/shap_xgb_dependence_top4.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved: shap_xgb_dependence_top4.{pdf,png}")


if __name__ == "__main__":
    main()
