"""WP3 extension (analysis only): do the new kernel shapes rank the WP3 reference errors higher?

For every WP3 test case (kernel_qc_report.TEST_SET, with the error cell's position), the whole-archive
rank of the best window that contains the error cell (whole hour if no cell is given), for four Isolation
Forest models per product:
  3x3 base   = 4 half-splits            3x3 shapes = + spike + 4 stripes
  4x4 base   = 4 half-splits            4x4 shapes = + patch
Rank 1 = most anomalous window of that model. Must-not cases (real front, Bora jet, ...) should not move up.
The Christmas 2025 whole-day case is skipped (no single hour).

Needs: kernel_features.py --window 3 4 --variant shapes, kernel_clustering.py --window 3 4 --variant shapes.

Outputs:
  outputs/reports/kernel_shapes_compare.csv
  outputs/figures/kernel_shapes_new.png      +/- patterns of the new shapes (report figure)

Run: python scripts/kernel_shapes_compare.py [--figure-only]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from kernel_clustering import REPORTS_DIR, load_models
from kernel_features import PRODUCTS, feature_path, make_shape_kernels
from kernel_plots import FIGURES_DIR, _draw_kernel_pattern
from kernel_qc_report import TEST_SET

MODELS = [("3x3 base", 3, ""), ("3x3 shapes", 3, "shapes"), ("4x4 base", 4, ""), ("4x4 shapes", 4, "shapes")]
SHAPE_TITLES = {"SPK": "SPK spike:\ncentre − 8 neighbours", "STH": "STH stripe —:\ncentre row − rows N/S",
                "STV": "STV stripe |:\ncentre column − columns W/E", "STD": "STD stripe ↘:\nNW–SE diagonal − rest",
                "STU": "STU stripe ↗:\nSW–NE diagonal − rest", "PAT": "PAT patch (4×4):\ncentre 2×2 − ring of 12"}


def best_rank_containing(models, ti, ci, cj, n):
    """Best window at feature-time index ti containing grid cell (ci, cj) (any
    window of the hour if ci is None): its whole-archive rank (1 = most
    anomalous), rank as % of all windows, and how many windows contain the cell."""
    m = models["t_idx"] == ti
    if ci is not None:
        wi, wj = models["wi"].astype(int), models["wj"].astype(int)
        m = m & (wi <= ci) & (ci < wi + n) & (wj <= cj) & (cj < wj + n)
    if not m.any():
        return None
    best = models["score"][m].max()
    rank = int((models["score"] > best).sum()) + 1
    return {"rank": rank, "rank_pct": 100.0 * rank / len(models["score"]), "n_windows": int(m.sum())}


def compare():
    rows = []
    by_product = {}
    for case in TEST_SET:
        if len(case["time"]) <= 10:
            print(f"  skipped (whole day): {case['label']}")
            continue
        by_product.setdefault(case["product"], []).append(case)
    for product, cases in by_product.items():
        with xr.open_dataset(PRODUCTS[product][0]) as src:
            lat, lon = src["latitude"].values, src["longitude"].values
        with xr.open_dataset(feature_path(product, 3)) as f:
            times = f["time"].values
        models = {name: load_models(product, n, v) for name, n, v in MODELS}
        for case in cases:
            hit = np.nonzero(times == np.datetime64(case["time"]))[0]
            row = {"product": product, "time": case["time"], "label": case["label"], "expect": case["expect"]}
            if len(hit) == 0:
                print(f"  {product} {case['time']}: not in the kernel archive, skipped")
                continue
            ci = cj = None
            if case["lat"] is not None:
                ci, cj = int(np.abs(lat - case["lat"]).argmin()), int(np.abs(lon - case["lon"]).argmin())
            for name, n, _ in MODELS:
                r = best_rank_containing(models[name], int(hit[0]), ci, cj, n)
                row[f"{name} rank"] = r["rank"] if r else None
                row[f"{name} top%"] = r["rank_pct"] if r else None
            rows.append(row)
    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "kernel_shapes_compare.csv"
    df.to_csv(out, index=False, float_format="%.6g")
    print(df[["product", "time", "label", "expect"] + [f"{m} rank" for m, _, _ in MODELS]].to_string(index=False))
    print(f"Saved: {out}")
    return df


def plot_shapes():
    k3, k4 = make_shape_kernels(3), make_shape_kernels(4)
    shapes = [(name, w) for name, w in k3.items()] + [(name, w) for name, w in k4.items()]
    fig, axes = plt.subplots(1, len(shapes), figsize=(2.3 * len(shapes), 2.8), dpi=200)
    for ax, (name, w) in zip(axes, shapes):
        _draw_kernel_pattern(ax, w)
        ax.set_title(SHAPE_TITLES[name], fontsize=8)
    fig.suptitle("New kernel shapes: mean of the + cells − mean of the − cells", fontsize=10, x=0.02,
                 ha="left", y=1.04)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "kernel_shapes_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


EXTRA_EXAMPLES = [("tri1", "2024-03-03T19:30", "single-cell spike (reference)"),
                  ("total", "2024-10-01T17:00", "Bora jet with one error cell (must not, except the cell)")]


def shapemap_name(product, time_iso):
    return f"kernel_shapemaps_{product}_{time_iso[:16].replace(':', '')}.png"


def _error_cell(product, time_iso):
    for c in TEST_SET:
        if c["product"] == product and c["time"] == time_iso[:16] and c["lat"] is not None:
            return c["lat"], c["lon"]
    return None


def plot_shape_maps(product, time_iso, label):
    """Field map(s) with the error cell marked, plus every new shape's value at
    every window position (SPK + 4 stripes on 3x3 windows, PAT on 4x4), same
    layout and colour scale as the kernel_featuremaps figures. Caption: rank of
    the best window this hour (containing the error cell, if known) with the
    half-splits only and with the shapes added."""
    from cartopy import crs as ccrs
    from kernel_features import load_good_snapshot
    from kernel_plots import FEATURE_VMAX, _draw_field, _draw_radial_vectors, _map_axes, _radial_uv
    from plot_qc_flagged_field import _cell_edges
    snap = load_good_snapshot(product, time_iso)
    if snap is None:
        print(f"  {product} {time_iso}: not in archive, skipped")
        return
    ti = snap["t_archive"]
    channels = list(PRODUCTS[product][1])
    lat, lon = snap["lat"], snap["lon"]
    radial = product != "total"
    pc = 2 if radial else 1
    shapes = [(k, 3) for k in make_shape_kernels(3)] + [(k, 4) for k in make_shape_kernels(4)]
    nch = len(channels)
    block = [0.9] + [2.6] * nch                       # one shape = its +/- pattern + one map per channel
    fig = plt.figure(figsize=(16 if radial else 22, 10.5), dpi=120)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.6], hspace=0.18)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.10)
    bottom = outer[1].subgridspec(2, 3 * len(block), width_ratios=block * 3, wspace=0.08, hspace=0.22)
    ax0 = _map_axes(fig, top[0, 0], lat, lon)
    h, lab = _draw_field(ax0, snap, product, 0)
    fig.colorbar(h, ax=ax0, orientation="horizontal", shrink=0.8, pad=0.08, label=lab)
    ax0.set_title("RDVA (QC-good cells)" if radial else "current field (QC-good cells)", fontsize=10, loc="left")
    map_axes = [ax0]
    if radial:
        axv = _map_axes(fig, top[0, 1], lat, lon, left_labels=False)
        q = _draw_radial_vectors(axv, lat, lon, *_radial_uv(product, snap["time"][0]))
        fig.colorbar(q, ax=axv, orientation="horizontal", shrink=0.8, pad=0.08, label="Speed (m/s)")
        axv.set_title("velocity (QC-good cells)", fontsize=10, loc="left")
        map_axes.append(axv)
    cell = _error_cell(product, time_iso)
    ci = cj = None
    if cell is not None:
        ci, cj = int(np.abs(lat - cell[0]).argmin()), int(np.abs(lon - cell[1]).argmin())
        for ax in map_axes:
            ax.plot(lon[cj], lat[ci], marker="o", markersize=16, markerfacecolor="none", markeredgecolor="#ff1493",
                    markeredgewidth=2.2, transform=ccrs.PlateCarree(), zorder=6)
    caption = []
    for name, n, v in MODELS:
        r = best_rank_containing(load_models(product, n, v), ti, ci, cj, n)
        caption.append(f"{name}: " + (f"rank {r['rank']:,} (top {r['rank_pct']:.3g}%)" if r else "no window"))
    where = "containing the circled error cell" if cell else "of this hour"
    tax = fig.add_subplot(top[0, 1 if not radial else 2])
    tax.axis("off")
    tax.text(0.0, 1.0, f"Best window {where}\n(rank 1 = most anomalous, whole archive):\n" + "\n".join(caption),
             transform=tax.transAxes, fontsize=9.5, va="top")
    tax.text(0.0, 0.42,
             "Each coloured square below = one window placed at its centre\n(3×3 for SPK and the stripes, 4×4 for PAT).\n"
             "White: target and surroundings agree. Red: the + cells are\nfaster toward east (u) / north (v) / away "
             "from the station (RDVA)\nthan the − cells; blue: the opposite.\n"
             "Pink circle: the error cell from the WP3 reference set (if any).",
             transform=tax.transAxes, fontsize=9, va="top")
    files = {n: xr.open_dataset(feature_path(product, n, "shapes")).isel(time=ti) for n in (3, 4)}
    mesh = None
    for s_i, (k, n) in enumerate(shapes):
        row, col0 = divmod(s_i, 3)
        col0 *= len(block)
        f = files[n]
        we, ae = _cell_edges(f["win_lon"].values), _cell_edges(f["win_lat"].values)
        pax = fig.add_subplot(bottom[row, col0])
        _draw_kernel_pattern(pax, make_shape_kernels(n)[k])
        for c, ch in enumerate(channels):
            ax = _map_axes(fig, bottom[row, col0 + 1 + c], lat, lon, labels=False)
            mesh = ax.pcolormesh(we, ae, np.ma.masked_invalid(f[f"{k}_{ch}"].values), transform=ccrs.PlateCarree(),
                                 cmap="RdBu_r", vmin=-FEATURE_VMAX, vmax=FEATURE_VMAX, zorder=1.5)
            if c == 0:
                ax.set_title(SHAPE_TITLES[k].replace("\n", " "), fontsize=8.5, loc="left")
            if nch > 1:
                ax.text(0.03, 0.95, ch, transform=ax.transAxes, fontsize=10, weight="bold", va="top", zorder=7)
            if cell is not None:
                ax.plot(lon[cj], lat[ci], marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor="#ff1493", markeredgewidth=1.6, transform=ccrs.PlateCarree(), zorder=6)
    for f in files.values():
        f.close()
    pos = tax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 + 0.02, 0.9 * pos.width, 0.012])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("shape value = mean(+ cells) − mean(− cells), m/s", fontsize=9)
    fig.suptitle(f"{product.upper()} — new kernel shapes over the whole map at {time_iso[:16]} UTC ({label})",
                 fontsize=12, x=0.05, ha="left", y=0.93)
    out = FIGURES_DIR / shapemap_name(product, time_iso)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_examples():
    from kernel_size_compare import EXAMPLES
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cases = [(p, t, lab) for p, ex in EXAMPLES.items() for t, lab in ex] + EXTRA_EXAMPLES
    for product, iso, label in cases:
        plot_shape_maps(product, iso, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure-only", action="store_true", help="only draw the shapes figure")
    ap.add_argument("--maps", action="store_true", help="only draw the shape-map figures for the examples")
    args = ap.parse_args()
    if args.maps:
        plot_examples()
        return
    plot_shapes()
    if not args.figure_only:
        compare()


if __name__ == "__main__":
    main()
