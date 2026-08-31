"""Sklop 2 Task 3: identify known artifacts/anomalies the existing EU HFR Node
QC does not catch.

Outputs:
  outputs/reports/qc_gap_summary.csv   - per-test: % of currently-good data
                                          each gap test would additionally flag
  outputs/reports/qc_gap_monthly.csv   - monthly version of the same
  outputs/figures/qc_gap_spatial.png   - spatial flag-rate maps, gap tests
                                          + GDOP_QC side by side for comparison
  outputs/figures/qc_gap_monthly.png   - monthly flag-rate lines

Run: conda activate hfr-qc && python scripts/qc_gap_analysis.py
"""
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TOTAL_PATH = PROJECT_ROOT / "data" / "processed" / "total" / "hfr_nadr_total_2021_present_unified.nc"

COLOR_INK = "#0b0b0b"
CATEGORICAL = ["#256abf", "#c0392b", "#2f9e44", "#e8830f", "#7048a8", "#0c8599"]

EARTH_RADIUS_M = 6_371_000.0

# "Starting default" thresholds -- demonstrative only, not calibrated. See
# module docstring.
DIV_K = 5.0
VORT_K = 5.0
SPIKE_K = 5.0
NEIGH_ANGLE_DEG = 90.0
NEIGH_MIN_VALID = 4
NEIGH_COHERENCE_MIN = 0.7  # mean resultant length of neighbor unit vectors;
# the concrete measure of "neighbors agree with each other" (avoids flagging
# genuinely turbulent/incoherent regions as "inconsistent").
CLIM_K = 3.0  # mean +/- k*std, per-month climatology pooled across all years
# (simplified: not leave-one-year-out -- sufficient to demonstrate the gap).

NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

TEST_LABELS = {
    "DIV": "divergence",
    "VORT": "vorticity",
    "NEIGH": "neighbor_consistency",
    "SPIKE": "temporal_spike",
    "CLIM": "climatology",
}


def load_total():
    ds = xr.open_dataset(TOTAL_PATH)
    ds = ds.squeeze("depth", drop=True)
    u = ds["EWCT"].values.astype(np.float32)
    v = ds["NSCT"].values.astype(np.float32)
    qcflag = ds["QCflag"].values.astype(np.float32)
    time = pd.to_datetime(ds["time"].values)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    ds.close()
    return u, v, qcflag, time, lat, lon


def grid_spacing_meters(lat, lon):
    dlat_deg = float(np.diff(lat).mean())
    dlon_deg = float(np.diff(lon).mean())
    dy = EARTH_RADIUS_M * np.radians(dlat_deg)
    dx_per_lat = EARTH_RADIUS_M * np.cos(np.radians(lat)) * np.radians(dlon_deg)  # (lat,)
    return dy, dx_per_lat


def divergence_and_vorticity(u, v, lat, lon):
    """Centered-difference div=du/dx+dv/dy, vorticity=dv/dx-du/dy.

    NaN wherever a cardinal (N/S/E/W) neighbor is missing -- undefined, not
    zero. No plane-fit fallback here.
    """
    dy, dx_per_lat = grid_spacing_meters(lat, lon)
    shape = u.shape
    du_dx = np.full(shape, np.nan, dtype=np.float32)
    dv_dx = np.full(shape, np.nan, dtype=np.float32)
    du_dy = np.full(shape, np.nan, dtype=np.float32)
    dv_dy = np.full(shape, np.nan, dtype=np.float32)

    du_dx[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx_per_lat[None, :, None])
    dv_dx[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dx_per_lat[None, :, None])
    du_dy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)
    dv_dy[:, 1:-1, :] = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)

    div = du_dx + dv_dy
    vort = dv_dx - du_dy
    return div, vort


def robust_flag(field, k):
    """Flag |field| > median(|field|) + k*MAD(|field|), global threshold."""
    abs_field = np.abs(field)
    med = np.nanmedian(abs_field)
    mad = np.nanmedian(np.abs(abs_field - med))
    threshold = med + k * mad
    flagged = abs_field > threshold
    computable = ~np.isnan(field)
    return flagged, computable, threshold


def spike_test(u, v, k=SPIKE_K):
    """Per-cell |d(velocity)/dt| vs. that cell's own robust threshold."""
    du_dt = np.diff(u, axis=0)
    dv_dt = np.diff(v, axis=0)
    accel = np.hypot(du_dt, dv_dt)
    med = np.nanmedian(accel, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(accel - med), axis=0, keepdims=True)
    threshold = med + k * mad
    flagged = accel > threshold
    computable = ~np.isnan(accel)
    pad = np.zeros((1,) + flagged.shape[1:], dtype=bool)
    flagged_full = np.concatenate([pad, flagged], axis=0)
    computable_full = np.concatenate([pad, computable], axis=0)
    return flagged_full, computable_full


def neighbor_consistency(u, v, angle_threshold_deg=NEIGH_ANGLE_DEG,
                          min_valid=NEIGH_MIN_VALID, coherence_min=NEIGH_COHERENCE_MIN):
    """Flag cells whose direction deviates >angle_threshold_deg from the
    coherent mean direction of their up-to-8 neighbors."""
    speed = np.hypot(u, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(speed > 0, u / speed, np.nan).astype(np.float32)
        uy = np.where(speed > 0, v / speed, np.nan).astype(np.float32)

    t, ny, nx = u.shape
    sum_x = np.zeros((t, ny, nx), dtype=np.float32)
    sum_y = np.zeros((t, ny, nx), dtype=np.float32)
    count = np.zeros((t, ny, nx), dtype=np.int16)

    for di, dj in NEIGHBOR_OFFSETS:
        shifted_x = np.full((t, ny, nx), np.nan, dtype=np.float32)
        shifted_y = np.full((t, ny, nx), np.nan, dtype=np.float32)
        i0, i1 = max(0, -di), ny - max(0, di)
        j0, j1 = max(0, -dj), nx - max(0, dj)
        si0, si1 = max(0, di), ny + min(0, di)
        sj0, sj1 = max(0, dj), nx + min(0, dj)
        shifted_x[:, i0:i1, j0:j1] = ux[:, si0:si1, sj0:sj1]
        shifted_y[:, i0:i1, j0:j1] = uy[:, si0:si1, sj0:sj1]
        valid = ~np.isnan(shifted_x)
        sum_x += np.where(valid, shifted_x, 0)
        sum_y += np.where(valid, shifted_y, 0)
        count += valid

    safe_count = np.where(count > 0, count, 1)
    mean_x = sum_x / safe_count
    mean_y = sum_y / safe_count
    coherence = np.hypot(mean_x, mean_y)
    neighbor_angle = np.degrees(np.arctan2(mean_y, mean_x))
    center_angle = np.degrees(np.arctan2(uy, ux))
    dtheta = np.abs(((center_angle - neighbor_angle + 180) % 360) - 180)

    valid_center = ~np.isnan(ux)
    computable = (count >= min_valid) & (coherence >= coherence_min) & valid_center
    flagged = computable & (dtheta > angle_threshold_deg)
    return flagged, computable


def climatology_test(u, v, time, k=CLIM_K):
    speed = np.hypot(u, v)
    months = time.month.values
    flagged = np.zeros(speed.shape, dtype=bool)
    computable = ~np.isnan(speed)
    for m in range(1, 13):
        idx = months == m
        sub = speed[idx]
        mean = np.nanmean(sub, axis=0, keepdims=True)
        std = np.nanstd(sub, axis=0, keepdims=True)
        lo, hi = mean - k * std, mean + k * std
        flagged[idx] = ((sub < lo) | (sub > hi)) & ~np.isnan(sub)
    return flagged, computable


def summarize_test(name, flagged, computable, qcflag, time, threshold=None):
    good = qcflag == 1
    good_computable = good & computable
    n_good_computable = int(good_computable.sum())
    n_good_flagged = int((flagged & good_computable).sum())
    pct_of_good_flagged = 100.0 * n_good_flagged / n_good_computable if n_good_computable else float("nan")

    n_computable = int(computable.sum())
    pct_computable = 100.0 * n_computable / computable.size

    row = {
        "test": name,
        "threshold": threshold,
        "pct_grid_computable": pct_computable,
        "n_good_and_computable": n_good_computable,
        "n_good_additionally_flagged": n_good_flagged,
        "pct_of_good_additionally_flagged": pct_of_good_flagged,
    }

    month = pd.Series(time).dt.strftime("%Y-%m").values
    df = pd.DataFrame({"month": month})
    flat_good_computable = good_computable.reshape(good_computable.shape[0], -1)
    flat_flagged = flagged.reshape(flagged.shape[0], -1)
    df["n_good_computable"] = flat_good_computable.sum(axis=1)
    df["n_good_flagged"] = (flat_flagged & flat_good_computable).sum(axis=1)
    monthly = df.groupby("month", as_index=False)[["n_good_computable", "n_good_flagged"]].sum()
    monthly["pct_of_good_flagged"] = (
        100.0 * monthly["n_good_flagged"] / monthly["n_good_computable"].where(monthly["n_good_computable"] > 0)
    )
    monthly["test"] = name

    return row, monthly


def spatial_rate(flagged, computable, qcflag):
    good_computable = (qcflag == 1) & computable
    n_flagged = (flagged & good_computable).sum(axis=0)
    n_computable = good_computable.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = 100.0 * n_flagged / np.where(n_computable > 0, n_computable, np.nan)
    return rate


def gdop_qc_bad_rate(lat, lon):
    """Load GDOP_QC bad-fraction spatial map (flag 3/4 of 1-4) for comparison,
    same definition as Task 2's qc_spatial_Total.png."""
    ds = xr.open_dataset(TOTAL_PATH)
    da = ds["GDOP_QC"].squeeze("depth", drop=True)
    is_bad = da.isin([3, 4]).sum(dim="time")
    is_valid = da.isin([1, 2, 3, 4]).sum(dim="time")
    rate = (is_bad / is_valid.where(is_valid > 0)) * 100.0
    ds.close()
    return rate.values


def plot_spatial(maps, lat, lon, out_path):
    n = len(maps)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), dpi=150, squeeze=False)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    for i, (name, arr) in enumerate(maps.items()):
        ax = axes[i // ncols][i % ncols]
        vmax = np.nanpercentile(arr, 98) if np.isfinite(arr).any() else 1.0
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
        # GDOP_QC here is the same existing-QC result as Task 2's qc_spatial_Total.png
        # (see gdop_qc_bad_rate() docstring) -- keep it in Task 2's GnBu so it reads as
        # the known/existing reference, while the actual new gap-test panels stay Purples.
        cmap = "GnBu" if name.startswith("GDOP_QC") else "Purples"
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(name, fontsize=9, color=COLOR_INK, loc="left")
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of good data flagged")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(
        "Total: gap tests vs. GDOP_QC -- % of currently-'good' (QCflag=1) data flagged, full record",
        color=COLOR_INK, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_monthly(monthly_df, out_path):
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    tests = monthly_df["test"].unique()
    for i, test in enumerate(tests):
        s = monthly_df[monthly_df["test"] == test].sort_values("month")
        ax.plot(s["month"], s["pct_of_good_flagged"], label=TEST_LABELS.get(test, test),
                color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=1.4)
    ax.set_title("Total: monthly % of currently-'good' data each gap test would flag", loc="left",
                 color=COLOR_INK, fontsize=10)
    ax.set_ylabel("% of good data flagged", color=COLOR_INK)
    months = monthly_df["month"].unique()
    n_ticks = 12
    xticks = list(range(0, len(months), max(1, len(months) // n_ticks)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([months[i] for i in xticks], rotation=45, ha="right", color=COLOR_INK, fontsize=7)
    ax.legend(frameon=False, fontsize=8, ncol=len(tests), loc="upper center", bbox_to_anchor=(0.5, -0.3))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Total dataset...")
    u, v, qcflag, time, lat, lon = load_total()
    print(f"  shape: {u.shape}, time range: {time.min()} .. {time.max()}")

    summary_rows = []
    monthly_frames = []
    spatial_maps = {}

    print("Divergence + vorticity...")
    div, vort = divergence_and_vorticity(u, v, lat, lon)
    div_flag, div_computable, div_thr = robust_flag(div, DIV_K)
    row, monthly = summarize_test("DIV", div_flag, div_computable, qcflag, time, div_thr)
    summary_rows.append(row); monthly_frames.append(monthly)
    spatial_maps["divergence"] = spatial_rate(div_flag, div_computable, qcflag)

    vort_flag, vort_computable, vort_thr = robust_flag(vort, VORT_K)
    row, monthly = summarize_test("VORT", vort_flag, vort_computable, qcflag, time, vort_thr)
    summary_rows.append(row); monthly_frames.append(monthly)
    spatial_maps["vorticity"] = spatial_rate(vort_flag, vort_computable, qcflag)
    del div, vort

    print("Neighbor consistency...")
    neigh_flag, neigh_computable = neighbor_consistency(u, v)
    row, monthly = summarize_test("NEIGH", neigh_flag, neigh_computable, qcflag, time)
    summary_rows.append(row); monthly_frames.append(monthly)
    spatial_maps["neighbor_consistency"] = spatial_rate(neigh_flag, neigh_computable, qcflag)
    del neigh_flag, neigh_computable

    print("Temporal spike...")
    spike_flag, spike_computable = spike_test(u, v)
    row, monthly = summarize_test("SPIKE", spike_flag, spike_computable, qcflag, time)
    summary_rows.append(row); monthly_frames.append(monthly)
    spatial_maps["temporal_spike"] = spatial_rate(spike_flag, spike_computable, qcflag)
    del spike_flag, spike_computable

    print("Climatology...")
    clim_flag, clim_computable = climatology_test(u, v, time)
    row, monthly = summarize_test("CLIM", clim_flag, clim_computable, qcflag, time)
    summary_rows.append(row); monthly_frames.append(monthly)
    spatial_maps["climatology"] = spatial_rate(clim_flag, clim_computable, qcflag)
    del clim_flag, clim_computable

    print("GDOP_QC (for comparison)...")
    spatial_maps["GDOP_QC (existing)"] = gdop_qc_bad_rate(lat, lon)

    del u, v, qcflag
    gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORTS_DIR / "qc_gap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print(summary_df.to_string(index=False))

    monthly_df = pd.concat(monthly_frames, ignore_index=True)
    monthly_path = REPORTS_DIR / "qc_gap_monthly.csv"
    monthly_df.to_csv(monthly_path, index=False)
    print(f"Saved: {monthly_path}")

    # Cache raw plot inputs to disk so plotting can be retried/debugged without
    # re-running the (slow) numpy computation above.
    debug_npz = REPORTS_DIR / "_qc_gap_spatial_maps_debug.npz"
    np.savez(debug_npz, lat=lat, lon=lon, **{k.replace(" ", "_"): v for k, v in spatial_maps.items()})
    monthly_df.to_pickle(REPORTS_DIR / "_qc_gap_monthly_debug.pkl")
    print(f"Cached plot inputs: {debug_npz}")

    plot_spatial(spatial_maps, lat, lon, FIGURES_DIR / "qc_gap_spatial.png")
    plot_monthly(monthly_df, FIGURES_DIR / "qc_gap_monthly.png")


if __name__ == "__main__":
    main()
