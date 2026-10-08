"""A radial's QC-good EWCT/NSCT field drawn in the cluster-plot
convention (cluster_regimes.plot_cluster_centroid_vectors): length ~ speed,
RADIAL_SCALE, length capped at TOTAL_VMAX, plasma_r 0-0.42 m/s.
Used for the undetected-artefact examples IZOL 2024-04-08T23:00 and AURI 2022-01-02T11:00.
Usage: python scripts/plot_radial_vectors.py PRODUCT TIME
Output: outputs/figures/kernel_vectors_<product>_<time>.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.pyplot as plt, numpy as np, xarray as xr
from cluster_regimes import RADIAL_SCALE, SPEED_CMAP, TOTAL_VMAX
from kernel_features import PRODUCTS, apply_good_mask, PROJECT_ROOT

prod, t_iso = sys.argv[1], sys.argv[2]
ds = xr.open_dataset(PRODUCTS[prod][0])
ti = int(np.nonzero(ds["time"].values == np.datetime64(t_iso))[0][0])
s = ds.isel(time=ti).squeeze("depth")
qc = s["QCflag"].values
u_raw = np.ma.masked_invalid(apply_good_mask(s["EWCT"].values, qc))
v_raw = np.ma.masked_invalid(apply_good_mask(s["NSCT"].values, qc))
lat, lon = ds["latitude"].values, ds["longitude"].values
ds.close()

speed = np.ma.sqrt(u_raw**2 + v_raw**2)
f = np.ma.where(speed > 0, np.ma.where(speed > TOTAL_VMAX, TOTAL_VMAX, speed) / speed, 1.0)
lon2d, lat2d = np.meshgrid(lon, lat)
fig = plt.figure(figsize=(7, 6), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon.min() - 0.02, lon.max() + 0.02, lat.min() - 0.02, lat.max() + 0.02], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"),
               facecolor="#d8d4c8", edgecolor="#5a5a5a", linewidth=0.6, zorder=2)
gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#c9c7bf", linestyle="--")
gl.top_labels = gl.right_labels = False
q = ax.quiver(lon2d, lat2d, u_raw * f, v_raw * f, speed, transform=ccrs.PlateCarree(), cmap=SPEED_CMAP,
              scale=RADIAL_SCALE, width=0.004, headwidth=2.5, headlength=4, headaxislength=3.5, zorder=3)
q.set_clim(0, TOTAL_VMAX)
fig.colorbar(q, ax=ax, shrink=0.75, pad=0.03).set_label("Speed (m/s)")
ax.quiverkey(q, X=0.87, Y=0.06, U=TOTAL_VMAX, label=f"{TOTAL_VMAX:.2f} m/s", labelpos="N",
             coordinates="axes", fontproperties={"size": 8})
ax.set_title(f"{prod.upper()} radial current field (QC-good cells), {t_iso[:16].replace('T', ' ')} UTC",
             fontsize=10, loc="left")
out = PROJECT_ROOT / "outputs" / "figures" / f"kernel_vectors_{prod}_{t_iso[:16].replace(':', '')}.png"
fig.savefig(out, bbox_inches="tight")
print("Saved:", out)
