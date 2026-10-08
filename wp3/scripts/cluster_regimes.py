"""WP3: k-means regime clustering over the full 2021-present archive.

Clusters each product (Total, AURI, PIRA, TRI1, IZOL) independently, one
k-means run per k in 2..8, to see whether unsupervised clustering recovers
recognizable wind-driven circulation regimes (Bora/Scirocco/Tramontana) on
its own, and whether a divergence-dominated cluster emerges.

Sample = one timestamp's whole-domain snapshot (flattened grid). Total uses
EWCT+NSCT (u+v) jointly; each radial station uses RDVA only (not EWCT/NSCT
-- a radial's EWCT/NSCT are RDVA decomposed along each cell's fixed
bearing, not an independent measurement).

Outputs:
  outputs/reports/cluster_summary_<product>.csv
  outputs/figures/cluster_<product>_k<k>_cluster<c>.png

Run: python scripts/cluster_regimes.py
"""
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

GOOD_FLAGS = {1, 2}

TOTAL_PATH = PROJECT_ROOT / "data" / "processed" / "total" / "hfr_nadr_total_2021_present_unified.nc"
RADIAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "radials" / "{station}"
    / "hfr_nadr_{station_lower}_2021_present_unified.nc"
)
STATIONS = ["AURI", "PIRA", "TRI1", "IZOL"]

PRODUCTS = {"Total": (TOTAL_PATH, ["EWCT", "NSCT"])}
for _station in STATIONS:
    _path = Path(str(RADIAL_PATH).format(station=_station, station_lower=_station.lower()))
    PRODUCTS[_station] = (_path, ["RDVA"])

MIN_COVERAGE = 0.75
K_VALUES = range(2, 9)
RANDOM_STATE = 42
SILHOUETTE_SAMPLE_SIZE = 5000

# Fixed color range, shared across EVERY cluster plot (Total and all 4
# radials alike): 0.27 m/s is the same color whether it's a Total or a
# radial plot, not a product-relative shade. TOTAL_VMAX (0.42) is the 95th
# percentile of max-speed-per-cluster across Total's own 35 clusters --
# the true max (0.43) comes from one atypically strong cluster; using it as
# the ceiling would compress every other cluster into a narrow dark band.
# Clusters above TOTAL_VMAX just saturate at the brightest color.
TOTAL_VMAX = 0.42

# Arrow LENGTH is CAPPED at TOTAL_VMAX regardless of a cluster's true speed
# (color still saturates at the right color), so no extreme outlier cluster
# (e.g. PIRA's, ~0.48 m/s) can run off the map. Radials get a larger
# per-unit length (RADIAL_SCALE) so weak radial clusters stay legible;
# both groups share the same cap.
TOTAL_SCALE = 8.0
RADIAL_SCALE = TOTAL_SCALE * (0.27 / TOTAL_VMAX)

# Yellow (weak) -> purple (strong). Plain `plasma_r`: both endpoints are far
# from white, so arrows stay visible on the white ocean background.
SPEED_CMAP = "plasma_r"


def build_feature_matrix(path, variables, min_coverage=MIN_COVERAGE):
    """Builds the (n_kept_timestamps, n_features) feature matrix for one
    product. A cell only counts as valid if QCflag is good AND the raw
    value is non-NaN. A timestamp qualifies ("kept") if its count of valid
    cells is >= min_coverage * (that product's own 99th-percentile valid
    count) -- relative to the product's own typical footprint, not the
    full 440-cell grid, since a single radial station's footprint never
    fills it. Among kept timestamps, any grid cell that is NEVER valid is
    dropped entirely (cell_mask=False) -- such a cell is structurally
    outside this product's range and has no observations to mean-impute
    from. Remaining per-cell gaps are filled with that cell's own mean
    over the kept timestamps where it WAS valid.
    """
    ds = xr.open_dataset(path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    n_lat, n_lon = len(lat), len(lon)

    qc = ds["QCflag"].squeeze("depth")
    good = qc.isin(list(GOOD_FLAGS)).values  # (n_time, n_lat, n_lon)

    channel_arrays = []
    for var in variables:
        arr = ds[var].squeeze("depth").values  # (n_time, n_lat, n_lon)
        arr = np.where(good & np.isfinite(arr), arr, np.nan)
        channel_arrays.append(arr.reshape(arr.shape[0], -1))  # (n_time, n_grid)

    valid_mask = good.reshape(good.shape[0], -1)  # (n_time, n_grid)
    n_valid = valid_mask.sum(axis=1)
    ref_full = float(np.percentile(n_valid, 99))
    coverage = n_valid / ref_full
    keep_time = coverage >= min_coverage

    times = ds["time"].values[keep_time]
    valid_mask_kept = valid_mask[keep_time]           # (n_kept, n_grid)
    cell_mask = valid_mask_kept.any(axis=0)            # (n_grid,)

    channels_kept = []
    for arr in channel_arrays:
        sub = arr[keep_time][:, cell_mask].copy()      # (n_kept, n_kept_cells)
        col_mean = np.nanmean(sub, axis=0)
        nan_rows, nan_cols = np.where(np.isnan(sub))
        sub[nan_rows, nan_cols] = col_mean[nan_cols]
        channels_kept.append(sub)

    X = np.hstack(channels_kept)  # (n_kept, n_kept_cells * len(variables))

    # Radial-only: each cell's DRVA (bearing from station to cell) is fixed
    # over time (sub-degree jitter only), so a single
    # per-cell mean over kept timestamps is "the" bearing. Needed to turn a
    # radial centroid's RDVA value back into plottable u/v arrows later --
    # not used for clustering itself (that still runs on RDVA alone).
    bearing = None
    if "RDVA" in variables:
        drva = ds["DRVA"].squeeze("depth").values.reshape(ds.sizes["time"], -1)  # (n_time, n_grid)
        bearing = np.nanmean(drva[keep_time][:, cell_mask], axis=0)  # (n_kept_cells,)

    ds.close()

    return {
        "times": times,
        "X": X,
        "cell_mask": cell_mask,
        "lat": lat,
        "lon": lon,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_channels": len(variables),
        "ref_full": ref_full,
        "bearing": bearing,
    }


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std_safe
    return X_scaled, mean, std_safe


def run_kmeans_sweep(X_scaled):
    results = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(
            X_scaled, labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, X_scaled.shape[0]),
            random_state=RANDOM_STATE,
        )
        results[k] = {
            "labels": labels,
            "centroids_scaled": km.cluster_centers_,
            "inertia": km.inertia_,
            "silhouette": sil,
        }
        print(f"  k={k}: inertia={km.inertia_:.1f} silhouette={sil:.3f}")
    return results


def _expand_to_grid(values, cell_mask, n_lat, n_lon):
    """Places a (n_kept_cells,) array of per-cell values back onto the full
    (n_lat, n_lon) grid -- cells dropped during feature-matrix building come
    back as NaN, same convention as a true coverage gap."""
    full = np.full(n_lat * n_lon, np.nan)
    full[cell_mask] = values
    return full.reshape(n_lat, n_lon)


def centroid_to_grid(centroid_scaled, mean, std, cell_mask, n_lat, n_lon, n_channels):
    """Inverse-transforms one cluster centroid back to physical units and
    splits it into `n_channels` full (n_lat, n_lon) grids."""
    centroid_raw = centroid_scaled * std + mean
    channels = np.split(centroid_raw, n_channels)
    return [_expand_to_grid(ch, cell_mask, n_lat, n_lon) for ch in channels]


def radial_centroid_to_uv_grid(centroid_scaled, mean, std, bearing, cell_mask, n_lat, n_lon):
    """Un-scales a radial product's (RDVA-only) centroid and decomposes it
    into u/v grids via each cell's fixed DRVA bearing -- same formula the
    archive's own EWCT/NSCT were computed with: east = RDVA*sin(bearing), north = RDVA*cos(bearing), degrees
    from North. The arrow DIRECTION this produces is fixed geometry (each
    cell's bearing to the station, same in every cluster) -- only the sign
    (toward/away) and magnitude vary between clusters."""
    rdva_raw = centroid_scaled * std + mean  # (n_kept_cells,)
    bearing_rad = np.deg2rad(bearing)
    u = rdva_raw * np.sin(bearing_rad)
    v = rdva_raw * np.cos(bearing_rad)
    return _expand_to_grid(u, cell_mask, n_lat, n_lon), _expand_to_grid(v, cell_mask, n_lat, n_lon)


def plot_cluster_centroid_vectors(u_grid, v_grid, lat, lon, product_name, k, cluster_id, cluster_pct, out_dir):
    u_raw = np.ma.masked_invalid(u_grid)
    v_raw = np.ma.masked_invalid(v_grid)
    speed = np.ma.sqrt(u_raw**2 + v_raw**2)
    lon2d, lat2d = np.meshgrid(lon, lat)
    pad_lon, pad_lat = 0.02, 0.02
    extent = [lon.min() - pad_lon, lon.max() + pad_lon, lat.min() - pad_lat, lat.max() + pad_lat]

    # Length scales with speed but is CAPPED at TOTAL_VMAX: a cluster faster
    # than TOTAL_VMAX is drawn at exactly the TOTAL_VMAX length. Color is fed
    # the true speed and saturates via q.set_clim; only the length is clamped.
    vmax = TOTAL_VMAX
    scale = TOTAL_SCALE if product_name == "Total" else RADIAL_SCALE
    length_speed = np.ma.where(speed > vmax, vmax, speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        length_scale_factor = np.ma.where(speed > 0, length_speed / speed, 1.0)
    u = u_raw * length_scale_factor
    v = v_raw * length_scale_factor

    fig = plt.figure(figsize=(7, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        facecolor="#d8d4c8", edgecolor="#5a5a5a", linewidth=0.6, zorder=2,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#c9c7bf", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Per-group scale fixed (not auto-scaled per plot) so a given speed is
    # the same arrow length in every figure within that group (see the
    # TOTAL_SCALE/RADIAL_SCALE comment above for why two groups). Color fed
    # the TRUE (uncapped) speed -- q.set_clim below makes it saturate at
    # the right color for anything above vmax, independent of the length
    # cap above. Arrow heads smaller than matplotlib's defaults (3/5/4.5).
    q = ax.quiver(
        lon2d, lat2d, u, v, speed,
        transform=ccrs.PlateCarree(), cmap=SPEED_CMAP,
        scale=scale, width=0.004, headwidth=2.5, headlength=4, headaxislength=3.5, zorder=3,
    )
    q.set_clim(0, vmax)
    cb = fig.colorbar(q, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("Speed (m/s)")
    ax.quiverkey(
        q, X=0.87, Y=0.06, U=vmax, label=f"{vmax:.2f} m/s",
        labelpos="N", coordinates="axes", fontproperties={"size": 8},
    )
    ax.set_title(
        f"{product_name} cluster {cluster_id} of k={k} mean field ({cluster_pct:.1f}% of kept timestamps)",
        fontsize=10, loc="left",
    )
    out_path = out_dir / f"cluster_{product_name.lower()}_k{k}_cluster{cluster_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def summarize_clusters(times, sweep, product_name, out_dir):
    """One row per (k, cluster): size, % of that product's kept
    timestamps, inertia/silhouette for that k (repeated per cluster row,
    for easy filtering), and month-of-year occurrence (Jan=1..Dec=12) as a
    %, to check for the winter-skew a real Bora regime would show."""
    rows = []
    months_of_year = pd.DatetimeIndex(times).month
    for k, res in sweep.items():
        labels = res["labels"]
        n = len(labels)
        for c in range(k):
            mask = labels == c
            size = int(mask.sum())
            moy_frac = pd.Series(months_of_year[mask]).value_counts(normalize=True).sort_index() * 100.0
            row = {
                "product": product_name,
                "k": k,
                "cluster": c,
                "size": size,
                "pct_of_kept": 100.0 * size / n,
                "inertia": res["inertia"],
                "silhouette": res["silhouette"],
            }
            for month in range(1, 13):
                row[f"month_{month:02d}_pct"] = float(moy_frac.get(month, 0.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    out_path = out_dir / f"cluster_summary_{product_name.lower()}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


# Known real events (WP2), used as an interpretability sanity check (not a formal validation metric): if a
# bora timestamp consistently lands in a cluster whose mean field looks
# like a coherent NE-to-SW flow, that's a strong signal the clustering is
# picking up real regimes. Total only.
ANCHOR_EVENTS_TOTAL = [
    ("2025-12-25T11:00:00", "bora, 25 Dec 2025 (well-covered, clean case)"),
    ("2024-10-05T02:00:00", "bora, 5 Oct 2024 (more modest coverage)"),
    ("2025-11-08T07:00:00", "uncaught Total divergence/convergence example"),
]


def report_anchor_clusters(times, sweep, product_name):
    if product_name != "Total":
        return
    print(f"\n-- [{product_name}] anchor-event cluster assignments --")
    for ts_str, desc in ANCHOR_EVENTS_TOTAL:
        ts = np.datetime64(ts_str)
        idx = np.where(times == ts)[0]
        if len(idx) == 0:
            print(f"  {desc} ({ts_str}): NOT in kept timestamps (dropped by the coverage filter)")
            continue
        i = int(idx[0])
        assignments = ", ".join(f"k={k}->cluster{res['labels'][i]}" for k, res in sweep.items())
        print(f"  {desc} ({ts_str}): {assignments}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for product_name, (path, variables) in PRODUCTS.items():
        print(f"\n=== [{product_name}] building feature matrix ===")
        feat = build_feature_matrix(path, variables)
        print(
            f"[{product_name}] n_kept={len(feat['times'])} "
            f"n_features={feat['X'].shape[1]} "
            f"cells_kept={feat['cell_mask'].sum()}/{len(feat['cell_mask'])}"
        )

        X_scaled, mean, std = standardize(feat["X"])

        print(f"[{product_name}] running k-means sweep k=2..8")
        sweep = run_kmeans_sweep(X_scaled)

        summarize_clusters(feat["times"], sweep, product_name, REPORTS_DIR)
        report_anchor_clusters(feat["times"], sweep, product_name)

        for k, res in sweep.items():
            for c in range(k):
                pct = 100.0 * (res["labels"] == c).sum() / len(res["labels"])
                if product_name == "Total":
                    u_grid, v_grid = centroid_to_grid(
                        res["centroids_scaled"][c], mean, std, feat["cell_mask"],
                        feat["n_lat"], feat["n_lon"], feat["n_channels"],
                    )
                else:
                    u_grid, v_grid = radial_centroid_to_uv_grid(
                        res["centroids_scaled"][c], mean, std, feat["bearing"], feat["cell_mask"],
                        feat["n_lat"], feat["n_lon"],
                    )
                plot_cluster_centroid_vectors(
                    u_grid, v_grid, feat["lat"], feat["lon"], product_name, k, c, pct, FIGURES_DIR,
                )


if __name__ == "__main__":
    main()
