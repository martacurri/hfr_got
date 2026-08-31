"""Sklop 2 Task 2: quantitative analysis of the existing EU HFR Node QC flags.

Outputs:
  outputs/reports/qc_flag_summary.csv   - overall histogram table
  outputs/reports/qc_flag_monthly.csv   - monthly bad-fraction table
  outputs/figures/qc_monthly_<product>.png   - monthly bad-fraction lines
  outputs/figures/qc_spatial_<product>.png   - spatial heatmaps, one per test

Run: conda run -n hfr-qc python scripts/qc_flag_analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
# Per-test color, assigned by name (not list position), so a test keeps
# the same color everywhere and pairs that sit close/overlap on the radial
# plots get maximally different hues rather than neighboring ones:
#   - QCflag vs RDCT_QC (the composite flag tracks RDCT_QC closely, so their
#     lines often overlap) -> blue vs red, opposite ends of the wheel.
#   - OWTR_QC vs VART_QC (both small-amplitude, values sit close together)
#     -> yellow vs violet, near-complementary.
# DDNS_QC/GDOP_QC (Total-only) reuse MDFL_QC/AVRB_QC's hues from the radial
# set — safe because Total and radial tests are never plotted on the same
# figure, so there's no on-chart collision.
TEST_COLORS = {
    "QCflag": "#2a78d6",    # blue
    "OWTR_QC": "#eda100",   # yellow
    "CSPD_QC": "#1baf7a",   # aqua
    "VART_QC": "#4a3aa7",   # violet
    "MDFL_QC": "#e87ba4",   # magenta
    "AVRB_QC": "#008300",   # green
    "RDCT_QC": "#e34948",   # red
    "DDNS_QC": "#eb6834",   # orange
    "GDOP_QC": "#e87ba4",   # magenta (reused; never co-plotted with MDFL_QC)
}

# ARGO flag scale (JERICO-NEXT D5.14 Appendix B).
FLAG_MEANINGS = {
    0: "no_qc_performed",
    1: "good_data",
    2: "probably_good_data",
    3: "potentially_correctable_bad_data",
    4: "bad_data",
    5: "value_changed",
    6: "not_used",
    7: "nominal_value",
    8: "interpolated_value",
    9: "missing_value",
}
GOOD_FLAGS = {1, 2}
BAD_FLAGS = {3, 4}
VALID_FLAGS = GOOD_FLAGS | BAD_FLAGS  # excludes 0/5-9, the rare/inapplicable codes

RADIAL_TESTS = ["QCflag", "OWTR_QC", "CSPD_QC", "VART_QC", "MDFL_QC", "AVRB_QC", "RDCT_QC"]
TOTAL_TESTS = ["QCflag", "CSPD_QC", "VART_QC", "DDNS_QC", "GDOP_QC"]
# POSITION_QC omitted from both: coordinate-check composite, no expected spatial pattern.

RADIAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "radials" / "{station}"
    / "hfr_nadr_{station_lower}_2021_present_unified.nc"
)
TOTAL_PATH = PROJECT_ROOT / "data" / "processed" / "total" / "hfr_nadr_total_2021_present_unified.nc"

STATIONS = ["AURI", "PIRA", "TRI1", "IZOL"]


def product_paths():
    paths = {"Total": (TOTAL_PATH, TOTAL_TESTS)}
    for station in STATIONS:
        p = Path(str(RADIAL_PATH).format(station=station, station_lower=station.lower()))
        paths[station] = (p, RADIAL_TESTS)
    return paths


def flag_histogram(da):
    """Counts per ARGO flag value plus a 'no_data' bucket for structural NaN."""
    values = da.values
    total_cells = values.size
    nan_mask = np.isnan(values)
    n_no_data = int(nan_mask.sum())
    counts = {"no_data": n_no_data}
    valid = values[~nan_mask].astype(int)
    for code, meaning in FLAG_MEANINGS.items():
        counts[meaning] = int((valid == code).sum())
    counts["_total_cells"] = total_cells
    return counts


def summarize_all(paths):
    rows = []
    for product, (path, tests) in paths.items():
        print(f"[{product}] opening {path.name}")
        ds = xr.open_dataset(path)
        for test in tests:
            print(f"  histogram: {test}")
            counts = flag_histogram(ds[test])
            n_valid = sum(counts[FLAG_MEANINGS[c]] for c in VALID_FLAGS)
            n_bad = sum(counts[FLAG_MEANINGS[c]] for c in BAD_FLAGS)
            pct_bad_of_valid = 100.0 * n_bad / n_valid if n_valid else float("nan")
            pct_missing = 100.0 * counts["missing_value"] / (counts["_total_cells"] - counts["no_data"]) \
                if (counts["_total_cells"] - counts["no_data"]) else float("nan")
            pct_no_data = 100.0 * counts["no_data"] / counts["_total_cells"]
            rows.append({
                "product": product,
                "test": test,
                **counts,
                "pct_bad_of_valid": pct_bad_of_valid,
                "pct_missing_of_attempted": pct_missing,
                "pct_no_data_of_grid": pct_no_data,
            })
        ds.close()
    return pd.DataFrame(rows)


def monthly_bad_fraction(paths):
    """Monthly fraction of BAD_FLAGS among VALID_FLAGS cells, per product x test."""
    frames = []
    for product, (path, tests) in paths.items():
        print(f"[{product}] monthly bad-fraction")
        ds = xr.open_dataset(path)
        month = ds["time"].dt.strftime("%Y-%m").rename("month")
        for test in tests:
            da = ds[test]
            is_bad = da.isin(list(BAD_FLAGS))
            is_valid = da.isin(list(VALID_FLAGS))
            bad_by_month = is_bad.groupby(month).sum(dim=[d for d in da.dims])
            valid_by_month = is_valid.groupby(month).sum(dim=[d for d in da.dims])
            frac = (bad_by_month / valid_by_month.where(valid_by_month > 0)) * 100.0
            df = frac.to_dataframe(name="pct_bad")[["pct_bad"]].reset_index()
            df = df.rename(columns={"month": "time"})
            df["product"] = product
            df["test"] = test
            frames.append(df)
        ds.close()
    return pd.concat(frames, ignore_index=True)


def spatial_bad_fraction(path, tests):
    """Per-grid-cell BAD_FLAGS fraction among VALID_FLAGS cells, across full record."""
    ds = xr.open_dataset(path)
    maps = {}
    for test in tests:
        da = ds[test]
        is_bad = da.isin(list(BAD_FLAGS)).sum(dim="time")
        is_valid = da.isin(list(VALID_FLAGS)).sum(dim="time")
        frac = (is_bad / is_valid.where(is_valid > 0)) * 100.0
        maps[test] = frac.squeeze()
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()
    return maps, lat, lon


def plot_monthly(monthly_df, product, out_path):
    sub = monthly_df[monthly_df["product"] == product]
    tests = sub["test"].unique()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for test in tests:
        s = sub[sub["test"] == test].sort_values("time")
        ax.plot(
            s["time"].astype(str), s["pct_bad"],
            label=test, color=TEST_COLORS[test], linewidth=1.4,
        )
    ax.set_title(f"{product}: monthly % flagged bad (of QC-evaluated cells)", loc="left", color=COLOR_INK, fontsize=10)
    ax.set_ylabel("% bad", color=COLOR_INK)
    n_ticks = 12
    xticks = list(range(0, len(sub["time"].unique()), max(1, len(sub["time"].unique()) // n_ticks)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([sub["time"].unique()[i] for i in xticks], rotation=45, ha="right", color=COLOR_INK, fontsize=7)
    ax.legend(frameon=False, fontsize=7, ncol=len(tests), loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(True, color=COLOR_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spatial(maps, lat, lon, product, out_path):
    n = len(maps)
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), dpi=150, squeeze=False)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    for i, (test, arr) in enumerate(maps.items()):
        ax = axes[i // ncols][i % ncols]
        im = ax.imshow(
            # "GnBu": sequential, pale water-green -> deep blue.
            arr.values, origin="lower", extent=extent, aspect="auto",
            cmap="GnBu", vmin=0, vmax=np.nanpercentile(arr.values, 98) or 1,
        )
        ax.set_title(test, fontsize=9, color=COLOR_INK, loc="left")
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% bad")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{product}: spatial % flagged bad by test (full 2021-present record)", color=COLOR_INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = product_paths()

    summary = summarize_all(paths)
    summary_path = REPORTS_DIR / "qc_flag_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    monthly = monthly_bad_fraction(paths)
    monthly["time"] = monthly["time"].astype(str)
    monthly_path = REPORTS_DIR / "qc_flag_monthly.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"Saved: {monthly_path}")

    for product in paths:
        plot_monthly(monthly, product, FIGURES_DIR / f"qc_monthly_{product}.png")

    for product, (path, tests) in paths.items():
        print(f"[{product}] spatial maps")
        maps, lat, lon = spatial_bad_fraction(path, tests)
        plot_spatial(maps, lat, lon, product, FIGURES_DIR / f"qc_spatial_{product}.png")


if __name__ == "__main__":
    main()
