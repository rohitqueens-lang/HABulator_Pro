"""
make_fig1_studyarea.py
======================
Figure 1 — study area + data overview for the Habulator paper.

(a) Great Lakes map (Albers Equal-Area, EPSG:5070) with the 73 EPA-GLNPO/GLENDA
    monitoring stations, coloured by sample count (viridis colourbar) at a uniform
    marker size; real lake polygons + state/province borders (Natural Earth 10m).
    Scale bar + N arrow.
(b) Raw biovolume time series for all five groups (2001–2021): individual samples
    (points, raw un-winsorized values) with the annual median per group (lines), on a
    LOG y-axis (mg/L) for display — biovolume spans ~5 orders of magnitude, so a linear
    axis collapses the variation. Shows temporal coverage + inter-annual dynamics;
    2020 absent (no EPA survey).
(c) Per-group empirical CDF on a LOG10 axis (spreads the ~5 orders of magnitude; honest
    about zero-inflation — left endpoint = zero fraction), paired with a shared-axis
    boxplot strip (median / IQR / 5–95%). This is the scale the emulator is trained on.

Real geometry in ./geo (downloaded once). Output: fig1_studyarea.{pdf,png}
"""
import warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import geopandas as gpd
warnings.filterwarnings("ignore")

LAKES = ["Superior", "Michigan", "Huron", "Erie", "Ontario"]
GROUPS = ["EDIAT", "LDIAT", "CHLOR", "CRYPT", "CYANO"]
GCOL = {"EDIAT": "#0072B2", "LDIAT": "#E69F00", "CHLOR": "#009E73",
        "CRYPT": "#CC79A7", "CYANO": "#D55E00"}  # Okabe–Ito (colorblind-safe), consistent across figures

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
})

# ── Data ──
d = pd.read_csv("habulator_master_v2.csv")
st = (d.groupby("station_norm")
        .agg(lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"),
             lake=("LAKE", "first"), n=("LATITUDE", "size")).reset_index())

# ── Geometry (Natural Earth 10m), bbox-subset then Albers (EPSG:5070) ──
def load(z):
    g = gpd.read_file(z)
    if g.crs is None:
        g = g.set_crs(4326)
    return g.to_crs(4326).cx[-92.5:-75.0, 41.0:49.3].to_crs(5070)
lakes = load("geo/lakes.zip")
states = load("geo/states.zip")
pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(5070)
pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y

fig = plt.figure(figsize=(7.2, 8.5))
gs = GridSpec(3, 1, height_ratios=[2.6, 1.05, 1.05], hspace=0.30)

# ══ (a) MAP ══
axm = fig.add_subplot(gs[0])
states.plot(ax=axm, color="none", edgecolor="#c2c8d0", lw=0.4, zorder=1)
lakes.plot(ax=axm, facecolor="#d6e8f5", edgecolor="#8fb4d6", lw=0.35, zorder=2)
sc = axm.scatter(pts.x, pts.y, c=pts["n"].to_numpy(), cmap="viridis", s=24,
                 edgecolors="#333", linewidths=0.3, alpha=0.95, zorder=5)
# lake labels at station centroids
for lk in LAKES:
    m = pts.lake == lk
    axm.text(pts.x[m].mean(), pts.y[m].mean(), f"Lake\n{lk}", ha="center", va="center",
             fontsize=7.2, fontweight="bold", color="#10324f", zorder=6,
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
# country orientation labels (north/south, centred to avoid the corner elements)
axm.text(0.5, 0.965, "CANADA", transform=axm.transAxes, ha="center", fontsize=6.5,
         color="#8a93a0", style="italic", zorder=6)
axm.text(0.5, 0.028, "UNITED STATES", transform=axm.transAxes, ha="center", fontsize=6.5,
         color="#8a93a0", style="italic", zorder=6)
# extent
b = pts.total_bounds; padx = (b[2]-b[0])*0.08; pady = (b[3]-b[1])*0.12
axm.set_xlim(b[0]-padx, b[2]+padx); axm.set_ylim(b[1]-pady, b[3]+pady)
axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
for s in axm.spines.values(): s.set_edgecolor("#888"); s.set_linewidth(0.6)
# scale bar (200 km) + N arrow
x0, y0 = axm.get_xlim()[0] + padx*0.5, axm.get_ylim()[0] + pady*0.6
axm.plot([x0, x0+200000], [y0, y0], color="#222", lw=1.6, solid_capstyle="butt", zorder=7)
axm.text(x0+100000, y0+12000, "200 km", ha="center", fontsize=6.3, color="#222")
# north arrow — top-right corner (away from labels/legend)
axm.annotate("N", xy=(0.965, 0.92), xytext=(0.965, 0.80),
             xycoords="axes fraction", textcoords="axes fraction",
             ha="center", fontsize=8.5, fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color="#222", lw=1.3))
axm.set_title("a", loc="left", fontweight="bold", fontsize=9.5)
# colourbar: sample count per station (encodes effort by colour, not size) — lower-right
cax = axm.inset_axes([0.70, 0.12, 0.27, 0.024])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cax.tick_params(labelsize=5.3, length=2, pad=1)
cax.xaxis.set_label_position("top")
cb.set_label("samples / station", fontsize=6.2, labelpad=2)

# ══ (b) Raw biovolume time series — all five groups, 2001–2021 ══
axts = fig.add_subplot(gs[1])
dt = pd.to_datetime(d["SAMPLING_DATE"]); yr = dt.dt.year
yspan = range(int(yr.min()), int(yr.max()) + 1)
for g in GROUPS:
    yv = d[g].to_numpy(float)
    axts.scatter(dt, np.where(yv > 0, yv, np.nan), s=2.4, color=GCOL[g],
                 alpha=0.18, linewidths=0, zorder=2)                               # individual samples (raw)
    med = d.assign(_y=yr).groupby("_y")[g].median().reindex(yspan)                 # NaN for 2020 → line breaks
    xt = pd.to_datetime([f"{y}-07-01" for y in med.index])
    axts.plot(xt, med.to_numpy(), color=GCOL[g], lw=1.4, marker="o", ms=2.8,
              mec="white", mew=0.4, label=g, zorder=4)                             # annual median
axts.set_yscale("log"); axts.set_ylim(8e-5, 1.6e2)                                 # log axis for display (raw values)
axts.set_ylabel("Biovolume (mg/L)"); axts.set_xlabel("Year")
axts.set_title("b", loc="left", fontweight="bold", fontsize=9.5)
axts.legend(ncol=5, frameon=False, fontsize=6.2, loc="upper center",
            bbox_to_anchor=(0.5, 1.05), handlelength=1.2, columnspacing=1.1, handletextpad=0.4)
axts.text(0.995, 0.03, "raw values · points: samples · lines: annual median · log axis",
          transform=axts.transAxes, ha="right", va="bottom", fontsize=5.8, color="#888")
for sp in ("top", "right"): axts.spines[sp].set_visible(False)

# ══ (c) Per-group distribution on the LOG scale — ECDF + familiar boxplot companion ══
# A true log axis spreads the ~5 orders of magnitude that panel (b) compresses. The ECDF
# is bandwidth-free and honest about zero-inflation (left endpoint = zero fraction); the
# shared-axis boxplot strip on top is the familiar five-number summary, so they read
# hand-in-hand. This is the scale on which the emulator is trained (log1p).
sub = gs[2].subgridspec(2, 1, height_ratios=[0.60, 1.0], hspace=0.08)
axbox = fig.add_subplot(sub[0])
axc = fig.add_subplot(sub[1], sharex=axbox)
XLEFT = 8e-5

# -- companion: horizontal boxplots of detected (>0) values, one row per group --
for i, g in enumerate(GROUPS):
    xp = d[g].dropna().to_numpy(float); xp = xp[xp > 0]
    bp = axbox.boxplot(xp, vert=False, positions=[i], widths=0.58, whis=(5, 95),
                       showfliers=False, patch_artist=True, manage_ticks=False, zorder=3)
    bp["boxes"][0].set(facecolor=GCOL[g], edgecolor=GCOL[g], alpha=0.85, lw=0.6)
    bp["medians"][0].set(color="white", lw=1.2)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=GCOL[g], lw=0.8)
axbox.set_xscale("log")
axbox.set_yticks(range(len(GROUPS))); axbox.set_yticklabels(GROUPS, fontsize=6.2)
for t, g in zip(axbox.get_yticklabels(), GROUPS):
    t.set_color(GCOL[g]); t.set_fontweight("bold")
axbox.set_ylim(-0.6, len(GROUPS) - 0.4); axbox.invert_yaxis()
axbox.tick_params(labelbottom=False, length=0)
axbox.set_title("c", loc="left", fontweight="bold", fontsize=9.5)
axbox.text(0.995, 1.04, "boxes: median · IQR · 5–95% of detected values",
           transform=axbox.transAxes, ha="right", va="bottom", fontsize=5.7, color="#888")
for sp in ("top", "right", "left"): axbox.spines[sp].set_visible(False)

# -- main: empirical CDF (shares the x-axis with the boxplots above) --
axc.axhline(0.5, color="#cccccc", lw=0.6, ls=(0, (4, 3)), zorder=1)  # median guide
axc.text(XLEFT * 1.4, 0.515, "median", fontsize=5.6, color="#9a9a9a", va="bottom", ha="left")
for g in GROUPS:
    x = np.sort(d[g].dropna().to_numpy(float))
    F = np.arange(1, len(x) + 1) / len(x)
    pos = x > 0
    axc.step(x[pos], F[pos], where="post", color=GCOL[g], lw=1.4, zorder=3)
    x0, f0 = x[pos][0], F[pos][0]
    axc.plot([x0], [f0], "o", ms=2.6, color=GCOL[g], zorder=4)          # curve start
    if (x == 0).mean() > 0.03:                                          # zero-inflated groups
        axc.plot([XLEFT, x0], [f0, f0], color=GCOL[g], lw=0.7, ls=":", zorder=2)
axc.set_xscale("log")
axc.set_xlim(XLEFT, 1.3e2); axc.set_ylim(0, 1.005)
axc.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
axc.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axc.set_xlabel("Biovolume (mg/L)  —  log scale"); axc.set_ylabel("Cumulative fraction")
for sp in ("top", "right"): axc.spines[sp].set_visible(False)

fig.savefig("figures/fig1_studyarea.pdf", bbox_inches="tight")
fig.savefig("figures/fig1_studyarea.png", bbox_inches="tight")
print(f"stations={len(st)} samples={len(d)}  per-lake:\n{st.groupby('lake').n.agg(['size','sum'])}")
print("Saved: fig1_studyarea.{pdf,png}")
