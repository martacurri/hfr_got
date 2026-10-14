"""WP3 extension: compare kernel window sizes 3/4/5/6.

Do larger windows (5x5, 6x6) find structures
that 3x3 misses, and does an anomaly's score fade or hold with window size?
(Hypothesis: a one-cell artefact is strongest at 3x3 and is averaged away in
larger windows; a real front or patch holds or grows.)

Needs the 5x5/6x6 feature files and models first:
  python scripts/kernel_features.py --window 5 6
  python scripts/kernel_clustering.py --window 5 6

Outputs:
  outputs/reports/kernel_sizes_coverage.csv   valid windows / hours per product and size
  outputs/reports/kernel_sizes_examples.csv   per example and size: best window, score, rank, box centre
  outputs/figures/kernel_sizes_<product>_<time>.png        (--figures) field + best box per size
  outputs/figures/kernel_featuremaps56_<product>_<time>.png (--figures) kernel maps for 5x5 and 6x6

Run: python scripts/kernel_size_compare.py [--coverage] [--examples] [--figures]   (no option = all three)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from kernel_clustering import REPORTS_DIR, load_models
from kernel_features import PRODUCTS, feature_path, load_good_snapshot
from kernel_plots import (BOX_COLOURS, COLOUR_NAMES, FIGURES_DIR, _draw_field, _draw_radial_vectors, _map_axes,
                          _outline, _radial_uv, plot_feature_maps)

SIZES = [3, 4, 5, 6]
# The 15 examples (3 per product, from the WP3 reference sets).
EXAMPLES = {
    "total": [("2023-02-02T20:00", "div/conv (reference)"), ("2024-10-22T08:00", "divergence (reference)"),
              ("2024-02-02T08:00", "shear (reference)")],
    "auri": [("2022-03-29T11:30", "shear"), ("2026-02-25T01:00", "shear"),
             ("2021-08-14T07:00", "far-range band artefact (contrast)")],
    "pira": [("2022-02-23T17:00", "divergence"), ("2024-08-08T01:30", "divergence"),
             ("2025-07-09T20:30", "shear")],
    "tri1": [("2024-10-22T03:30", "target divergence (reference)"), ("2023-11-03T04:00", "real front (must not)"),
             ("2024-06-11T17:00", "TRI1-only artefact")],
    "izol": [("2024-06-28T08:00", "in-field target (reference)"),
             ("2024-07-08T03:30", "in-field patch diluted by 3x3"), ("2023-11-01T22:00", "divergence flag example")],
}


def best_window_at(models, ti):
    """Most anomalous window of one timestamp (feature-file time index ti) and
    its whole-archive rank (1 = most anomalous of this size). None if the hour
    has no complete window of this size."""
    m = models["t_idx"] == ti
    if not m.any():
        return None
    r = int(np.nonzero(m)[0][np.argmax(models["score"][m])])
    s = models["score"][r]
    rank = int((models["score"] > s).sum()) + 1
    return {"wi": int(models["wi"][r]), "wj": int(models["wj"][r]), "score": s, "rank": rank,
            "rank_pct": 100.0 * rank / len(models["score"]), "n_windows_hour": int(m.sum())}


def coverage_row(t_idx, n_times):
    n_hours = len(np.unique(t_idx))
    return {"n_windows": int(len(t_idx)), "n_hours": n_hours, "pct_hours": 100.0 * n_hours / n_times}


def size_rows(product, ti, models_by_size, win_coords):
    """One row per window size for one example hour; empty fields where the
    size has no complete window at that hour."""
    rows = []
    for n, models in models_by_size.items():
        b = best_window_at(models, ti)
        row = {"product": product, "size": n, "score": None, "rank": None, "rank_pct": None,
               "box_lat": None, "box_lon": None, "n_windows_hour": 0, "n_windows_total": len(models["score"])}
        if b is not None:
            wlat, wlon = win_coords[n]
            row.update(score=float(b["score"]), rank=b["rank"], rank_pct=b["rank_pct"],
                       box_lat=float(wlat[b["wi"]]), box_lon=float(wlon[b["wj"]]),
                       n_windows_hour=b["n_windows_hour"], wi=b["wi"], wj=b["wj"])
        rows.append(row)
    return rows


def _times_and_coords(product, n):
    with xr.open_dataset(feature_path(product, n)) as f:
        return f["time"].values, f["win_lat"].values, f["win_lon"].values


def write_coverage():
    rows = []
    for product in PRODUCTS:
        base = None
        for n in SIZES:
            times, _, _ = _times_and_coords(product, n)
            r = coverage_row(load_models(product, n)["t_idx"], len(times))
            base = base or r["n_hours"]
            rows.append({"product": product, "size": n, **r, "hours_vs_3x3_pct": 100.0 * r["n_hours"] / base})
    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "kernel_sizes_coverage.csv"
    df.to_csv(out, index=False, float_format="%.2f")
    print(df.to_string(index=False))
    print(f"Saved: {out}")


def example_table():
    """All 15 examples x 4 sizes. Returns the DataFrame (also used by the figures)."""
    rows = []
    for product, examples in EXAMPLES.items():
        models = {n: load_models(product, n) for n in SIZES}
        coords = {}
        for n in SIZES:
            times, wlat, wlon = _times_and_coords(product, n)
            coords[n] = (wlat, wlon)
        for iso, label in examples:
            hit = np.nonzero(times == np.datetime64(iso))[0]  # same time axis for every size
            if len(hit) == 0:
                print(f"  {product} {iso}: not in the kernel archive, skipped")
                continue
            for row in size_rows(product, int(hit[0]), models, coords):
                rows.append({"time": iso, "label": label, **row})
    return pd.DataFrame(rows)


def write_examples():
    df = example_table()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "kernel_sizes_examples.csv"
    df.to_csv(out, index=False, float_format="%.6g")  # 6 significant digits: tiny rank %s stay readable
    print(df[["product", "time", "size", "score", "rank", "rank_pct", "box_lat", "box_lon"]].to_string(index=False))
    print(f"Saved: {out}")
    return df


def format_top_pct(pct):
    """Two significant digits, so 0.000063% doesn't print as 0.000%."""
    return f"{float(f'{pct:.2g}'):f}".rstrip("0").rstrip(".") + "%"


def box_draw_order(rows):
    """Sizes that have a window, largest first, so the smaller boxes are drawn
    on top and stay visible when they sit inside a larger one."""
    have = rows[rows["score"].notna()]
    return sorted((int(n) for n in have["size"]), reverse=True)


def plot_size_overview(product, iso, label, rows):
    """Field map(s) with the most anomalous window of each size as a box, and a
    per-size table (score, whole-archive rank, box centre). rows = this
    example's rows of example_table()."""
    snap = load_good_snapshot(product, iso)
    if snap is None:
        print(f"  {product} {iso}: not in archive, skipped")
        return
    radial = product != "total"
    fig = plt.figure(figsize=(14 if radial else 9, 8.5), dpi=130)
    gs = fig.add_gridspec(2, 2 if radial else 1, height_ratios=[1, 0.32], hspace=0.12, wspace=0.08)
    ax0 = _map_axes(fig, gs[0, 0], snap["lat"], snap["lon"])
    h, lab = _draw_field(ax0, snap, product, 0)
    fig.colorbar(h, ax=ax0, orientation="horizontal", shrink=0.8, pad=0.06, label=lab)
    axes = [ax0]
    if radial:
        axv = _map_axes(fig, gs[0, 1], snap["lat"], snap["lon"], left_labels=False)
        q = _draw_radial_vectors(axv, snap["lat"], snap["lon"], *_radial_uv(product, snap["time"][0]))
        fig.colorbar(q, ax=axv, orientation="horizontal", shrink=0.8, pad=0.06, label="Speed (m/s)")
        axes.append(axv)
    by_size = {int(r["size"]): r for _, r in rows.iterrows()}
    for n in box_draw_order(rows):
        for ax in axes:
            _outline(ax, snap, int(by_size[n]["wi"]), int(by_size[n]["wj"]), n, BOX_COLOURS[n])
    lines = []
    for _, r in rows.sort_values("size").iterrows():
        n = int(r["size"])
        if pd.isna(r["score"]):
            lines.append(f"{n}×{n} ({COLOUR_NAMES[n]}): no complete window at this hour")
            continue
        lines.append(f"{n}×{n} ({COLOUR_NAMES[n]}): score {r['score']:.3f}, rank {int(r['rank']):,} of "
                     f"{int(r['n_windows_total']):,} (top {format_top_pct(r['rank_pct'])}), box centre "
                     f"{r['box_lat']:.3f}N {r['box_lon']:.3f}E")
    tax = fig.add_subplot(gs[1, :])
    tax.axis("off")
    tax.text(0.0, 1.0, "Most anomalous window of each size at this hour "
             "(rank 1 = most anomalous window of that size, whole archive):\n" + "\n".join(lines),
             fontsize=9.5, va="top", family="monospace", transform=tax.transAxes)
    fig.suptitle(f"{product.upper()} {iso} UTC — {label}: window sizes 3×3 to 6×6", fontsize=12, x=0.05,
                 ha="left", y=0.95)
    out = FIGURES_DIR / f"kernel_sizes_{product}_{iso.replace(':', '')}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_examples(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for product, examples in EXAMPLES.items():
        models56 = {n: load_models(product, n) for n in (5, 6)}
        for iso, label in examples:
            rows = df[(df["product"] == product) & (df["time"] == iso)]
            if rows.empty:
                continue
            plot_size_overview(product, iso, label, rows)
            plot_feature_maps(product, iso + ":00", label, models56, sizes=[5, 6])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    run_all = not (args.coverage or args.examples or args.figures)
    df = None
    if args.coverage or run_all:
        write_coverage()
    if args.examples or run_all:
        df = write_examples()
    if args.figures or run_all:
        plot_examples(df if df is not None else example_table())


if __name__ == "__main__":
    main()
