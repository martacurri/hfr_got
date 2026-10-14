"""Persistently bad grid cells, all products.

Which cells are always or almost always unusable, per product (Total + the 4
radial stations), counted only over the hours the product is really running
(>= RUN_MIN_GOOD QC-good cells), so station outages don't count against a cell:

  A  almost always bad: reports in >= 5% of running hours, existing QC flags >= 90% of its values
  B  often bad:         same, flagged 50-90%
  C  rarely usable:     has QC-good data, but in < 1% of running hours (slips through QC, almost
                        never there, e.g. the Total NE-corner cell 45.770N 13.568E)
  (A wins over C. A "category D" -- passes QC but persistently inconsistent with its
   neighbours -- is left for later.)

Known cases this must find: 45.648N 13.742E (Trieste, TRI1 + IZOL, OWTR_QC land mask) and
45.540N 13.645E (Izola, TRI1).

Station positions: WP1 station table.

Outputs:
  outputs/reports/persistent_bad_cells.csv   one row per flagged cell and product
  outputs/figures/persistent_bad_cells_map.png  one gulf map, each cell split into 5 strips
                                               (Total, AURI, PIRA, TRI1, IZOL)
  outputs/reports/persistent_bad_cells_by_period.csv, outputs/figures/persistent_bad_cells_by_period.png
      the same for two date periods: 2021-01-01..2023-02-28 (only AURI + PIRA) and
      2023-06-01..2026-07-31 (all four stations installed)

Run: python scripts/persistent_bad_cells.py
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from kernel_features import GOOD_FLAGS, PRODUCTS, PROJECT_ROOT
from plot_qc_flagged_field import _cell_edges

RUN_MIN_GOOD = 20       # a product "runs" in an hour with at least this many QC-good cells
MIN_REPORT = 0.05       # A/B need the cell to report in >= 5% of running hours
BAD_A, BAD_B = 0.90, 0.50
RARE_C = 0.01
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
PRODUCT_COLOURS = {"total": "#222222", "auri": "#1f77b4", "pira": "#2ca02c", "tri1": "#ff7f0e", "izol": "#9467bd"}
# WP1 station table: lat, lon
STATIONS = {"auri": (45.741, 13.669), "pira": (45.530, 13.566), "tri1": (45.6583, 13.7605),
            "izol": (45.5454, 13.6866)}
VALUE_VAR = {"total": "EWCT", "auri": "RDVA", "pira": "RDVA", "tri1": "RDVA", "izol": "RDVA"}


def run_hours(good):
    """good: (T, ny, nx) bool. True for hours with >= RUN_MIN_GOOD good cells."""
    return good.reshape(len(good), -1).sum(axis=1) >= RUN_MIN_GOOD


def classify_cells(raw, good, bad, n_on):
    """Per cell: 'A', 'B', 'C' or ''. raw/good/bad = counts over running hours."""
    raw, good, bad = (np.asarray(a, dtype=float) for a in (raw, good, bad))
    with np.errstate(invalid="ignore", divide="ignore"):
        bad_share = np.where(raw > 0, bad / raw, 0.0)
    reports = raw / n_on >= MIN_REPORT
    cat = np.full(raw.shape, "", dtype="<U1")
    cat[(good > 0) & (good / n_on < RARE_C)] = "C"
    cat[reports & (bad_share >= BAD_B)] = "B"
    cat[reports & (bad_share >= BAD_A)] = "A"
    return cat


def period_filter(times, start, end):
    """True for times from start 00:00 to end 23:59 (dates as 'YYYY-MM-DD')."""
    return np.asarray((times >= pd.Timestamp(start)) & (times < pd.Timestamp(end) + pd.Timedelta(days=1)))


def cell_counts(product, period=None):
    """Counts over the product's running hours; period = (start, end) dates
    restricts them to that date range."""
    path = PRODUCTS[product][0]
    ds = xr.open_dataset(path)
    x = ds[VALUE_VAR[product]].squeeze("depth").values
    qc = ds["QCflag"].squeeze("depth").values
    times = pd.DatetimeIndex(ds["time"].values)
    raw = np.isfinite(x)
    good = raw & np.isin(qc, GOOD_FLAGS)
    on = run_hours(good)
    if period is not None:
        on = on & period_filter(times, *period)
    out = {"lat": ds["latitude"].values, "lon": ds["longitude"].values, "n_on": int(on.sum()),
           "n_hours": len(x), "raw": raw[on].sum(0), "good": good[on].sum(0), "bad": (raw & ~good)[on].sum(0)}
    ds.close()
    return out


def table(counts):
    rows = []
    for product, c in counts.items():
        cat = classify_cells(c["raw"], c["good"], c["bad"], c["n_on"])
        for i, j in zip(*np.nonzero(cat != "")):
            raw = c["raw"][i, j]
            rows.append({"product": product, "category": cat[i, j], "lat": round(float(c["lat"][i]), 3),
                         "lon": round(float(c["lon"][j]), 3), "row": int(i), "col": int(j),
                         "running_hours": c["n_on"], "reports_pct": 100 * raw / c["n_on"],
                         "flagged_bad_pct": 100 * c["bad"][i, j] / raw if raw else np.nan,
                         "usable_pct": 100 * c["good"][i, j] / c["n_on"]})
    return pd.DataFrame(rows).sort_values(["category", "lat", "lon", "product"])


def draw_map(ax, df, lat, lon, stations, left_labels=True):
    """Grid, land, flagged-cell strips and the given stations on one axis."""
    le, ae = _cell_edges(lon), _cell_edges(lat)
    products = list(PRODUCT_COLOURS)
    ax.set_extent([le[0] - 0.01, le[-1] + 0.01, ae[0] - 0.01, ae[-1] + 0.01], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"), facecolor="#e6e3da",
                   edgecolor="#5a5a5a", linewidth=0.6, zorder=1)
    for x in le:
        ax.plot([x, x], [ae[0], ae[-1]], color="#d0cec7", linewidth=0.3, transform=ccrs.PlateCarree(), zorder=1.5)
    for y in ae:
        ax.plot([le[0], le[-1]], [y, y], color="#d0cec7", linewidth=0.3, transform=ccrs.PlateCarree(), zorder=1.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = left_labels
    for _, r in df.iterrows():
        k = products.index(r["product"])
        w = (le[r["col"] + 1] - le[r["col"]]) / len(products)
        x0, y0, h = le[r["col"]] + k * w, ae[r["row"]], ae[r["row"] + 1] - ae[r["row"]]
        colour = PRODUCT_COLOURS[r["product"]]
        style = {"A": dict(facecolor=colour, edgecolor=colour, linewidth=0.8),
                 "B": dict(facecolor="none", edgecolor=colour, hatch="////", linewidth=0.8),
                 "C": dict(facecolor="none", edgecolor=colour, linewidth=1.4)}[r["category"]]
        ax.add_patch(Rectangle((x0, y0), w, h, transform=ccrs.PlateCarree(), zorder=3, **style))
    for product in stations:
        s_lat, s_lon = STATIONS[product]
        ax.plot(s_lon, s_lat, marker="^", markersize=10, color=PRODUCT_COLOURS[product], markeredgecolor="k",
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(s_lon + 0.006, s_lat + 0.004, product.upper(), fontsize=9, weight="bold",
                transform=ccrs.PlateCarree(), zorder=5)


def legend_handles():
    products = list(PRODUCT_COLOURS)
    handles = [Patch(facecolor=PRODUCT_COLOURS[p], label=p.upper() if p != "total" else "Total") for p in products]
    handles += [Patch(facecolor="#777777", label="A: flagged bad ≥90% of the time"),
                Patch(facecolor="none", edgecolor="#777777", hatch="////", label="B: flagged bad 50–90%"),
                Patch(facecolor="none", edgecolor="#777777", linewidth=1.4, label="C: usable in <1% of running hours"),
                Line2D([], [], marker="^", color="w", markerfacecolor="#999999", markeredgecolor="k", markersize=9,
                       label="station (WP1 coordinates)")]
    return handles


def plot_map(df, counts):
    c0 = next(iter(counts.values()))
    fig = plt.figure(figsize=(11, 9), dpi=170)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    draw_map(ax, df, c0["lat"], c0["lon"], list(STATIONS))
    ax.legend(handles=legend_handles(), loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_title("Persistently bad grid cells, 2021–2026: each cell split into 5 strips\n"
                 "(left to right: Total, AURI, PIRA, TRI1, IZOL); only hours when the product runs "
                 f"(≥{RUN_MIN_GOOD} good cells)", fontsize=10, loc="left")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "persistent_bad_cells_map.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# Station periods by date (WP1 station table; first data in our EU-node archive).
# IZOL's Oct 2022 - Feb 2023 test data is not official, and Total does not use it (its GDOP cells stay
# 100% flagged until March 2023). Mar - May 2023 (AURI + PIRA + IZOL, before TRI1's first data on
# 2023-06-01) is a short 3-station transition and is not shown.
PERIODS = [("Period 1: only AURI + PIRA", ("2021-01-01", "2023-02-28"), ["auri", "pira"]),
           ("Period 2: all four stations installed (AURI, PIRA, TRI1, IZOL)", ("2023-06-01", "2026-07-31"),
            ["auri", "pira", "tri1", "izol"])]


def plot_period_maps(counts):
    """Two panels by date: period 1 (only AURI + PIRA) and period 2 (all four
    stations installed). Each product counted over its own running hours in the
    period, outages included. Same categories as the whole-archive map."""
    c0 = next(iter(counts.values()))
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.6), dpi=150, subplot_kw={"projection": ccrs.PlateCarree()},
                             gridspec_kw={"wspace": 0.08})
    rows, panels = [], []
    for k, (ax, (name, (start, end), stations)) in enumerate(zip(axes, PERIODS)):
        sub = {p: cell_counts(p, (start, end)) for p in ["total", *stations]}
        df = table(sub)
        df.insert(0, "period", f"{name}, {start} to {end}")
        rows.append(df)
        draw_map(ax, df, c0["lat"], c0["lon"], stations, left_labels=(k == 0))
        ax.set_title(f"{name}\n{start} to {end}", fontsize=11, loc="left")
        panels.append((name, start, end, sub))
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.125, 0.97, "Persistently bad grid cells before and after the network extension (Total B cells = "
             "GDOP_QC); each cell split into 5 strips: Total, AURI, PIRA, TRI1, IZOL. "
             "Mar-May 2023 (3-station transition) not shown.", fontsize=11)
    out = FIGURES_DIR / "persistent_bad_cells_by_period.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    for name, start, end, sub in panels:
        print(f"{name} ({start} to {end}): running half-hours per product: "
              + ", ".join(f"{p} {c['n_on']:,}" for p, c in sub.items()))
    return pd.concat(rows, ignore_index=True)


def main():
    counts = {p: cell_counts(p) for p in PRODUCTS}
    for p, c in counts.items():
        print(f"{p}: running hours {c['n_on']:,} of {c['n_hours']:,}")
    df = table(counts)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "persistent_bad_cells.csv"
    pct = ["reports_pct", "flagged_bad_pct", "usable_pct"]
    df[pct] = df[pct].round(2)
    df.to_csv(out, index=False)  # lat/lon keep 3 decimals (grid step 0.0135 deg)
    print(df.to_string(index=False))
    print(f"Saved: {out}")
    plot_map(df, counts)
    per = plot_period_maps(counts)
    pct = ["reports_pct", "flagged_bad_pct", "usable_pct"]
    per[pct] = per[pct].round(2)
    out = REPORTS_DIR / "persistent_bad_cells_by_period.csv"
    per.to_csv(out, index=False)
    print(per.to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
