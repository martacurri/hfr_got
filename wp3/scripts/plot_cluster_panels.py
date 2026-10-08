"""All clusters of one Total k-means run as panels of one figure, same style as
cluster_regimes.plot_cluster_centroid_vectors (length ~ speed capped at TOTAL_VMAX, TOTAL_SCALE,
plasma_r 0-TOTAL_VMAX). Usage: python scripts/plot_cluster_panels.py K
Output: outputs/figures/cluster_total_k<K>_all.png (report figures for K=2 and K=6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from cluster_regimes import (FIGURES_DIR, PRODUCTS, RANDOM_STATE, SPEED_CMAP, TOTAL_SCALE, TOTAL_VMAX,
                             build_feature_matrix, centroid_to_grid, standardize)

k = int(sys.argv[1])
path, variables = PRODUCTS["Total"]
feat = build_feature_matrix(path, variables)
X, mean, std = standardize(feat["X"])
km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
lat, lon = feat["lat"], feat["lon"]
lon2d, lat2d = np.meshgrid(lon, lat)
extent = [lon.min() - 0.02, lon.max() + 0.02, lat.min() - 0.02, lat.max() + 0.02]

ncol = min(k, 3)
nrow = int(np.ceil(k / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.0 if nrow == 1 else 4.0 * nrow), dpi=200,
                         subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
q = None
for c in range(nrow * ncol):
    ax = axes.flat[c]
    if c >= k:
        ax.set_visible(False)
        continue
    u, v = centroid_to_grid(km.cluster_centers_[c], mean, std, feat["cell_mask"], feat["n_lat"], feat["n_lon"],
                            feat["n_channels"])
    u, v = np.ma.masked_invalid(u), np.ma.masked_invalid(v)
    speed = np.ma.sqrt(u**2 + v**2)
    f = np.ma.where(speed > 0, np.ma.where(speed > TOTAL_VMAX, TOTAL_VMAX, speed) / speed, 1.0)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"),
                   facecolor="#d8d4c8", edgecolor="#5a5a5a", linewidth=0.6, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#c9c7bf", linestyle="--")
    gl.top_labels = gl.right_labels = False
    if c % ncol:
        gl.left_labels = False
    q = ax.quiver(lon2d, lat2d, u * f, v * f, speed, transform=ccrs.PlateCarree(), cmap=SPEED_CMAP,
                  scale=TOTAL_SCALE, width=0.004, headwidth=2.5, headlength=4, headaxislength=3.5, zorder=3)
    q.set_clim(0, TOTAL_VMAX)
    ax.quiverkey(q, X=0.87, Y=0.06, U=TOTAL_VMAX, label=f"{TOTAL_VMAX:.2f} m/s", labelpos="N",
                 coordinates="axes", fontproperties={"size": 8})
    pct = 100.0 * (km.labels_ == c).mean()
    ax.set_title(f"Cluster {c} ({pct:.1f}% of timestamps)", fontsize=11, loc="left")
fig.colorbar(q, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, label="Speed (m/s)")
fig.canvas.draw()
top = max(ax.get_position().y1 for ax in axes.flat if ax.get_visible())
fig.suptitle(f"Total surface currents: k-means k={k}, mean field of each cluster", fontsize=13, x=0.06,
             ha="left", y=top + 0.09)
out = FIGURES_DIR / f"cluster_total_k{k}_all.png"
fig.savefig(out, bbox_inches="tight")
print("Saved:", out, "| cluster sizes:", [round(100 * (km.labels_ == c).mean(), 1) for c in range(k)])
