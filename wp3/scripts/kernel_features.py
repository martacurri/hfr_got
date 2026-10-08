"""WP3: sliding-window kernel features (the "convolution" step).

Window size 4 (files kernel_responses_<product>.nc) or 3 (..._3x3.nc), chosen
with --window (default both). For 3x3 the middle row / column / diagonal has
weight 0, so each kernel is a centred difference across the centre cell.
IZOL starts at 2023-03-01: its earlier test period is dropped (PRODUCT_START).

A 4x4 window slides one cell at a time over every timestamp's grid. At each
position the 4 half-split kernels each return
mean(half A) - mean(half B), signed, in m/s:

  TB  top/bottom   A = north 2 rows,  B = south 2 rows
  LR  left/right   A = east 2 cols,   B = west 2 cols
  DR  diagonal \\   A = north-east triangle above the NW->SE diagonal
  UR  diagonal /   A = north-west triangle above the SW->NE diagonal
  (diagonal cells themselves weigh 0)

Total: applied to u (EWCT) and v (NSCT) -> 8 features per window.
Radials: applied to RDVA -> 4 features per window.
A window is only valid if all n*n cells are QC-good; otherwise NaN.

Array orientation everywhere: row 0 = SOUTH (latitude increases with row
index), col 0 = WEST -- so "north" = higher row index, not row 0.

Run: python scripts/kernel_features.py [--window 4 3] [--products ...]
"""
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from numpy.lib.stride_tricks import sliding_window_view

KERNEL_NAMES = ["TB", "LR", "DR", "UR"]
WINDOW = 4
GOOD_FLAGS = (1, 2)
PRODUCT_START = {"izol": np.datetime64("2023-03-01")}  # IZOL test period before the official start (work_project.pdf)


def size_suffix(n):
    """File-name suffix per window size: 4x4 keeps the original names."""
    return "" if n == 4 else f"_{n}x{n}"


def source_start_index(times, product):
    """First source-time index kept for this product (0 = keep all)."""
    start = PRODUCT_START.get(product)
    return 0 if start is None else int(np.searchsorted(times, start))


def make_kernels(n=WINDOW):
    """Half-split kernels as (n, n) weight arrays in array orientation
    (row 0 = south). Signs come from centred indices; each half is then
    normalised so the result is mean(half A) - mean(half B)."""
    idx = np.arange(n) - (n - 1) / 2.0
    i = idx[:, None] + np.zeros((1, n))  # row offset, + = north
    c = idx[None, :] + np.zeros((n, 1))  # col offset, + = east
    signs = {
        "TB": np.sign(i),
        "LR": np.sign(c),
        "DR": np.sign(i + c),  # + = north-east of the NW->SE diagonal
        "UR": np.sign(i - c),  # + = north-west of the SW->NE diagonal
    }
    kernels = {}
    for name in KERNEL_NAMES:
        s = signs[name]
        w = np.zeros((n, n))
        w[s > 0] = 1.0 / (s > 0).sum()
        w[s < 0] = -1.0 / (s < 0).sum()
        kernels[name] = w
    return kernels


def apply_good_mask(values, qcflag):
    good = np.isin(qcflag, GOOD_FLAGS) & np.isfinite(values)
    return np.where(good, values, np.nan).astype(np.float32)


def kernel_responses(field, kernels):
    """field: (T, ny, nx), NaN = unusable cell. Returns {name: (T, ny-n+1,
    nx-n+1) float32}, NaN wherever any of the window's n*n cells is NaN."""
    n = next(iter(kernels.values())).shape[0]
    win = sliding_window_view(field, (n, n), axis=(1, 2))  # (T, wy, wx, n, n)
    complete = ~np.isnan(win).any(axis=(-2, -1))
    filled = np.nan_to_num(win, nan=0.0)  # NaNs are masked out below anyway
    out = {}
    for name, w in kernels.items():
        r = np.einsum("tyxab,ab->tyx", filled, w)
        out[name] = np.where(complete, r, np.nan).astype(np.float32)
    return out


def window_centres(coord, n=WINDOW):
    return sliding_window_view(np.asarray(coord, dtype=np.float64), n).mean(axis=-1)


def window_any(flag, n=WINDOW):
    """True for each window containing at least one True cell."""
    return sliding_window_view(flag, (n, n), axis=(1, 2)).any(axis=(-2, -1))


def spread_window_counts(counts, n=WINDOW):
    """Adds each window's count to all n*n grid cells the window covers."""
    wy, wx = counts.shape
    out = np.zeros((wy + n - 1, wx + n - 1))
    for a in range(n):
        for b in range(n):
            out[a:a + wy, b:b + wx] += counts
    return out


def adjacent_sign_flip_windows(field, min_abs, n=WINDOW):
    """True for each window containing two edge-adjacent cells of opposite
    sign, both with |value| >= min_abs (a genuine flip, not two near-zero
    values straddling 0). Used as the radial benchmark in validation."""
    a = np.abs(field) >= min_abs
    flip_h = (field[:, :, :-1] * field[:, :, 1:] < 0) & a[:, :, :-1] & a[:, :, 1:]
    flip_v = (field[:, :-1, :] * field[:, 1:, :] < 0) & a[:, :-1, :] & a[:, 1:, :]
    in_h = sliding_window_view(flip_h, (n, n - 1), axis=(1, 2)).any(axis=(-2, -1))
    in_v = sliding_window_view(flip_v, (n - 1, n), axis=(1, 2)).any(axis=(-2, -1))
    return in_h | in_v


def anomaly_rate_map(all_counts, top_counts, n=WINDOW, min_windows=1000):
    """Per grid cell: % of the complete windows covering it that are in the
    top set. NaN where fewer than min_windows complete windows cover the cell
    (too few to give a meaningful rate). Returns (rate, denom)."""
    denom = spread_window_counts(all_counts, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(denom >= min_windows, 100.0 * spread_window_counts(top_counts, n) / denom, np.nan)
    return rate, denom


def window_mean_speed(u, v, t_idx, wi, wj, n=WINDOW):
    """Mean current speed over each listed window's n*n cells."""
    off = np.arange(n)
    rows = wi[:, None, None] + off[None, :, None]
    cols = wj[:, None, None] + off[None, None, :]
    t = t_idx[:, None, None]
    return np.hypot(u[t, rows, cols], v[t, rows, cols]).mean(axis=(1, 2))


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "kernels"
_RADIAL = PROJECT_ROOT / "data" / "processed" / "radials" / "{S}" / "hfr_nadr_{s}_2021_present_unified.nc"

PRODUCTS = {"total": (PROJECT_ROOT / "data" / "processed" / "total" / "hfr_nadr_total_2021_present_unified.nc",
                      {"u": "EWCT", "v": "NSCT"})}
for _s in ["auri", "pira", "tri1", "izol"]:
    PRODUCTS[_s] = (Path(str(_RADIAL).format(S=_s.upper(), s=_s)), {"rdva": "RDVA"})

TIME_CHUNK = 5000  # timestamps per convolution batch (~100 MB of windows)


def feature_path(product, n=WINDOW):
    return FEATURE_DIR / f"kernel_responses_{product}{size_suffix(n)}.nc"


def load_good_fields(product):
    """Whole-archive QC-good-masked fields for one product (used by the
    plotting/validation scripts to draw real windows and fields)."""
    path, channels = PRODUCTS[product]
    ds = xr.open_dataset(path)
    ds = ds.isel(time=slice(source_start_index(ds["time"].values, product), None))
    qc = ds["QCflag"].squeeze("depth").values
    out = {ch: apply_good_mask(ds[var].squeeze("depth").values, qc) for ch, var in channels.items()}
    out.update(time=ds["time"].values, lat=ds["latitude"].values, lon=ds["longitude"].values)
    ds.close()
    return out


def load_good_snapshot(product, time_iso):
    """Same as load_good_fields, but for one timestamp only (fast), with a
    length-1 time axis. Returns None if the time isn't in the archive."""
    path, channels = PRODUCTS[product]
    ds = xr.open_dataset(path)
    i0 = source_start_index(ds["time"].values, product)
    hit = np.nonzero(ds["time"].values == np.datetime64(time_iso))[0]
    if len(hit) == 0 or hit[0] < i0:
        ds.close()
        return None
    snap = ds.isel(time=slice(int(hit[0]), int(hit[0]) + 1)).squeeze("depth")
    qc = snap["QCflag"].values
    out = {ch: apply_good_mask(snap[var].values, qc) for ch, var in channels.items()}
    out.update(time=snap["time"].values, lat=ds["latitude"].values, lon=ds["longitude"].values,
               t_archive=int(hit[0]) - i0)
    ds.close()
    return out

def write_product(product, n=WINDOW):
    path, channels = PRODUCTS[product]
    kernels = make_kernels(n)
    ds = xr.open_dataset(path)
    ds = ds.isel(time=slice(source_start_index(ds["time"].values, product), None))
    times = ds["time"].values
    lat, lon = ds["latitude"].values, ds["longitude"].values
    n_time = len(times)
    out_path = feature_path(product, n)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(out_path, "w") as nc:
        nc.createDimension("time", n_time)
        nc.createDimension("win_lat", len(lat) - n + 1)
        nc.createDimension("win_lon", len(lon) - n + 1)
        tv = nc.createVariable("time", "f8", ("time",))
        tv.units = "seconds since 1970-01-01 00:00:00"
        tv.calendar = "standard"
        tv[:] = times.astype("datetime64[s]").astype("int64").astype("float64")
        for name, coord, units in [("win_lat", lat, "degrees_north"), ("win_lon", lon, "degrees_east")]:
            cv = nc.createVariable(name, "f8", (name,))
            cv.units = units
            cv.long_name = f"centre of {n}x{n} window"
            cv[:] = window_centres(coord, n)
        nc_vars = {}
        for ch in channels:
            for k in KERNEL_NAMES:
                var = nc.createVariable(f"{k}_{ch}", "f4", ("time", "win_lat", "win_lon"),
                                        zlib=True, complevel=4, fill_value=np.float32(np.nan))
                var.units = "m/s"
                var.long_name = f"{k} half-split kernel on {ch}: mean(half A) - mean(half B)"
                nc_vars[f"{k}_{ch}"] = var
        nc.window_size = n
        nc.start_time_cut = str(PRODUCT_START.get(product, "none"))
        nc.source_file = path.name
        nc.description = ("Sliding-window half-split kernel responses (WP3). Window valid only "
                          "if all cells QC-good (QCflag 1/2) and finite; NaN otherwise. "
                          "Row 0 of the source grid = south.")

        n_complete = 0
        for t0 in range(0, n_time, TIME_CHUNK):
            t1 = min(n_time, t0 + TIME_CHUNK)
            chunk = ds.isel(time=slice(t0, t1)).squeeze("depth")
            qc = chunk["QCflag"].values
            for ch, src in channels.items():
                resp = kernel_responses(apply_good_mask(chunk[src].values, qc), kernels)
                for k in KERNEL_NAMES:
                    nc_vars[f"{k}_{ch}"][t0:t1] = resp[k]
            n_complete += int(np.isfinite(resp["TB"]).sum())
            print(f"  {product}: {t1}/{n_time}")
    ds.close()
    n_pos = n_time * (len(lat) - n + 1) * (len(lon) - n + 1)
    print(f"Saved: {out_path}  complete windows: {n_complete:,} ({100 * n_complete / n_pos:.1f}% of positions)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, nargs="+", default=[4, 3], help="window sizes (default: 4 3)")
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS), help="default: all")
    args = ap.parse_args()
    for n in args.window:
        for product in args.products:
            write_product(product, n)


if __name__ == "__main__":
    main()
