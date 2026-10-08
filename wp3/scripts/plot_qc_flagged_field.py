"""Surface-current-field snapshots with each grid cell's individual QC-test
failures drawn underneath the current-field quiver as a colored patch --
"where and what does each QC flag actually flag." Also provides the RDVA
colormap and cell-edge helper used by the WP3 kernel figures.

Self-contained (no import from the WP2 scripts): the constants it needs
(TEST_COLORS, per-product test lists, good/bad flag codes) are duplicated
from WP2's qc_flag_analysis.py.

Per-cell coloring rule:
  - Colored by the *individual* tests (OWTR/CSPD/VART/MDFL/AVRB/RDCT for
    radials; CSPD/VART/DDNS/GDOP for Total), never the composite QCflag --
    QCflag is just their union, so it can't say *which* test caught a cell.
    POSITION_QC is excluded (no expected spatial pattern).
  - A cell passing every individual test, or with no QC evaluation at all
    this timestep (QCflag NaN), is left uncolored.
  - A cell failing exactly one individual test gets that test's color
    (same TEST_COLORS as WP2, so a test keeps the same color everywhere).
  - A cell failing 2+ individual tests at once gets a neutral "multiple
    flags" color (dark gray).
  - Quiver arrows for EWCT/NSCT are drawn on top at every cell with real
    (non-NaN) velocity regardless of QC flag.

AVRB_QC is excluded from the per-flag showcase set (not from the coloring):
it never fires (0.0% bad) at any of the 4 radial stations, so no example
scan exists for it.

Four example sets per product (skipped where not applicable):
  1. Good/bad -- the 0%-bad scan (or TRI1's least-bad fallback) vs. the
     highest-%bad scan.
  2. Screenshot reproductions -- 6 known WP2 timestamps (total1/2, auri1/2,
     izol1/2).
  3. Diverse scan -- the timestep with the most distinct individual tests
     each failing a non-trivial number of cells at once, tie-broken by total
     bad-cell count.
  4. Per-flag showcase -- for each individual test, the timestep where it
     fails the most cells *exclusively* (no other individual test also bad
     on that same cell). Tests that never fire exclusively (CSPD_QC for
     radials, VART_QC/DDNS_QC for Total) are skipped with a message.
All four restricted to scans with substantial coverage (>= MIN_VALID_FRAC
of the 99th-percentile evaluated-cell count), so a near-empty scan can't
produce a misleadingly extreme percentage.

Outputs: outputs/figures/qc_flagged_field_<product>_<label>.png
Chosen timestamps are printed to stdout.

RDVA field reproductions (radials only; Total has no RDVA): for the 6
screenshot timestamps and IZOL's outage-window pick, a signed-RDVA per-cell
color field (`RdYlBu_r` centered at 0: red = away from station, blue =
toward), matching the HFR Node viewer. RDVA's sign reveals artefacts that
EWCT/NSCT arrows hide behind bearing geometry (a sign flip between cells
with near-parallel bearings has no geometric explanation). QC-bad cells get
a pink (`#ff1493`) hatch overlay.
Outputs: outputs/figures/qc_rdva_field_<product>_<label>.png

Run: python scripts/plot_qc_flagged_field.py
"""
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

GOOD_FLAGS = {1, 2}
BAD_FLAGS = {3, 4}
VALID_FLAGS = GOOD_FLAGS | BAD_FLAGS

MIN_VALID_FRAC = 0.7  # same "substantial coverage" definition as the WP2 field-example plots
MIN_DIVERSE_CELLS = 2  # a test must fail at least this many cells to "count" toward the diverse-scan ranking

TOTAL_PATH = PROJECT_ROOT / "data" / "processed" / "total" / "hfr_nadr_total_2021_present_unified.nc"
RADIAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "radials" / "{station}"
    / "hfr_nadr_{station_lower}_2021_present_unified.nc"
)
STATIONS = ["AURI", "PIRA", "TRI1", "IZOL"]

# Individual (non-composite) tests per product -- same sets as
# qc_flag_analysis.py, QCflag/POSITION_QC excluded (see module docstring).
RADIAL_TESTS = ["OWTR_QC", "CSPD_QC", "VART_QC", "MDFL_QC", "AVRB_QC", "RDCT_QC"]
TOTAL_TESTS = ["CSPD_QC", "VART_QC", "DDNS_QC", "GDOP_QC"]

FLAGS_NEVER_FIRE = {"AVRB_QC"}  # see module docstring

PRODUCTS = {"Total": (TOTAL_PATH, TOTAL_TESTS)}
for _station in STATIONS:
    PRODUCTS[_station] = (
        Path(str(RADIAL_PATH).format(station=_station, station_lower=_station.lower())),
        RADIAL_TESTS,
    )

# Same per-test colors as qc_flag_analysis.py's TEST_COLORS (duplicated, not
# imported -- see module docstring).
TEST_COLORS = {
    "OWTR_QC": "#eda100",
    "CSPD_QC": "#1baf7a",
    "VART_QC": "#4a3aa7",
    "MDFL_QC": "#e87ba4",
    "AVRB_QC": "#008300",
    "RDCT_QC": "#e34948",
    "DDNS_QC": "#eb6834",
    "GDOP_QC": "#e87ba4",
}
MULTI_COLOR = "#3a3a3a"  # dedicated color: cell fails 2+ individual tests at once
PATCH_ALPHA = 0.65  # baked into each color's own RGBA, not passed as pcolormesh's global
# alpha= -- a global alpha on a QuadMesh overrides every cell's own color alpha
# (matplotlib Collection behavior), which turned the transparent "none"/good
# color into translucent BLACK, visually indistinguishable from MULTI_COLOR.

# Arrows are normalized to unit length (see plot_qc_flagged_field), so this
# scale is tuned against a fixed magnitude of 1; speed is shown by the
# colorbar.
QUIVER_SCALE = {name: 22.0 for name in PRODUCTS}

KNOWN_EXAMPLES = [
    ("Total", "2021-01-01T00:00:00", "screenshot-total1"),
    ("Total", "2022-10-21T06:00:00", "screenshot-total2"),
    ("AURI", "2022-01-02T10:30:00", "screenshot-auri1"),
    ("AURI", "2022-01-02T11:00:00", "screenshot-auri2"),
    ("IZOL", "2025-08-14T17:00:00", "screenshot-izol1"),
    ("IZOL", "2025-02-21T00:00:00", "screenshot-izol2"),
]

# Named real-storm (bora) examples: does CSPD_QC
# reject genuine wind-driven extremes along with artefacts? Timestamps found
# by searching each named storm window in the Total product for the
# best-covered peak in domain-mean speed (not just highest speed at any
# single, possibly near-empty, cell -- same "substantial coverage" concern as
# the good/bad picks above). Total only ("the GOT view"), title_label names
# the storm so the two don't read as generic diverse/showcase picks.
#   - 2025-12-25 11:00: squarely the reported Christmas-2025 bora date, a
#     well-covered scan (195/~440 cells) with a genuinely coherent ~14h
#     elevated-speed period around it (not a single-step spike) -- domain-mean
#     1.05 m/s, CSPD_QC rejects 66.7% of the scan. The clean case: a real,
#     broad, sustained storm signature, about two-thirds rejected on
#     magnitude alone.
#   - 2024-10-05 02:00: storm reported on 3-4 Oct 2024; searching that window
#     plus one day either side, the best-covered/clearest CSPD_QC signal falls
#     on the 5th (228/~440 cells, CSPD_QC 7.9% bad); the 3rd-4th peaks were
#     either near-empty or dominated by GDOP_QC (geometry, unrelated).
BORA_EXAMPLES = [
    ("Total", "2025-12-25T11:00:00", "bora-2025-12-25", "bora, 25 Dec 2025"),
    ("Total", "2024-10-05T02:00:00", "bora-2024-10-05", "bora, 5 Oct 2024"),
]

# IZOL's RDCT_QC/QCflag near-total-failure episode (WP2): ~99-100% bad
# continuously from ~March 2025 through ~June 2026. A concrete look at
# where/what is flagged during this specific known event, not just the archive-wide "bad" pick
# (which could land anywhere in the archive).
OUTAGE_WINDOWS = {
    "IZOL": (np.datetime64("2025-03-01"), np.datetime64("2026-06-30")),
}

# Extra per-flag showcase examples for tests worth seeing more than once:
# GDOP_QC's seam is a persistent geometric feature (WP2's main Total
# finding), worth a few dates, not
# just its single worst timestep. Any test not listed here still gets 1.
N_SHOWCASE_EXAMPLES = {"GDOP_QC": 3}

# Same "different picks must be genuinely different events" rule as
# the WP2 field-example plots' MIN_GAP_DAYS.
MIN_GAP_DAYS = 30


def _pick_ranked(pool, order_values, times, count, min_gap_days=MIN_GAP_DAYS):
    """Greedily picks up to `count` indices from `pool`, best-first by
    order_values (descending), skipping any candidate within min_gap_days of
    an already-picked time -- so a second/third pick is a genuinely
    different event, not an adjacent timestep of the same one. Same logic as
    the WP2 field-example plots' helper of the same name."""
    ordered = pool[np.argsort(-order_values[pool], kind="stable")]
    gap = np.timedelta64(min_gap_days, "D")
    picked = []
    for idx in ordered:
        if all(abs(times[idx] - times[p]) >= gap for p in picked):
            picked.append(idx)
        if len(picked) == count:
            break
    return picked


def per_timestep_bad_fraction(ds, test):
    """Whole-grid valid/bad cell counts and % bad for one test, one row per timestep."""
    da = ds[test].squeeze("depth")
    dims = [d for d in da.dims if d != "time"]
    is_valid = da.isin(list(VALID_FLAGS))
    is_bad = da.isin(list(BAD_FLAGS))
    n_valid = is_valid.sum(dim=dims).values
    n_bad = is_bad.sum(dim=dims).values
    pct_bad = np.where(n_valid > 0, n_bad / np.maximum(n_valid, 1) * 100.0, np.nan)
    return n_valid, pct_bad


def substantial_coverage_mask(ds):
    n_valid, _ = per_timestep_bad_fraction(ds, test="QCflag")
    ref_full = np.percentile(n_valid, 99)
    return n_valid >= MIN_VALID_FRAC * ref_full


def pick_good_bad(ds, product_name):
    """Same selection algorithm as the WP2 field-example plots' rank-1
    picks: 0%-bad (most-complete-coverage tiebreak) vs. highest-%-bad, both
    restricted to substantial coverage; falls back to the archive-wide
    least-bad timestep if no 0%-bad one exists (TRI1)."""
    times = ds["time"].values
    n_valid, pct_bad = per_timestep_bad_fraction(ds, test="QCflag")
    substantial = substantial_coverage_mask(ds)

    bad_pool = np.where(substantial)[0]
    bad_idx = bad_pool[np.argmax(pct_bad[bad_pool])]
    print(f"[{product_name}] bad pick: {times[bad_idx]} ({pct_bad[bad_idx]:.1f}% bad)")

    good_mask = substantial & (pct_bad == 0)
    if good_mask.any():
        good_pool = np.where(good_mask)[0]
        good_idx = good_pool[np.argmax(n_valid[good_pool])]
    else:
        any_reporting = np.where(n_valid > 0)[0]
        good_idx = any_reporting[np.argmin(pct_bad[any_reporting])]
        print(f"[{product_name}] WARNING: no 0%-bad timestep anywhere in the archive -- using least-bad fallback.")
    print(f"[{product_name}] good pick: {times[good_idx]} ({pct_bad[good_idx]:.2f}% bad)")

    return times[good_idx], times[bad_idx]


def pick_diverse(ds, product_name, tests, time_window=None, require_multiple=True, tag="diverse"):
    """Timestep with the most distinct individual tests each failing >=
    MIN_DIVERSE_CELLS cells at once, tie-broken by total bad-cell count.
    `time_window`, if given as (start, end) datetime64s, restricts the
    candidate pool to that range first (e.g. a known outage window) --
    otherwise the full substantial-coverage pool across the whole archive is
    used. `require_multiple=False` returns the best pick even if it only
    ever involves a single test (used for the outage case, where "it's
    entirely one test" is itself the answer worth showing)."""
    times = ds["time"].values
    substantial = substantial_coverage_mask(ds)
    if time_window is not None:
        start, end = time_window
        substantial = substantial & (times >= start) & (times <= end)
    pool = np.where(substantial)[0]
    if len(pool) == 0:
        print(f"[{product_name}] {tag}: no substantial-coverage timestep found in the requested window, skipped.")
        return None

    n_tests_involved = np.zeros(len(times), dtype=int)
    total_bad = np.zeros(len(times), dtype=int)
    for test in tests:
        da = ds[test].squeeze("depth")
        dims = [d for d in da.dims if d != "time"]
        bad_count = da.isin(list(BAD_FLAGS)).sum(dim=dims).values
        n_tests_involved += (bad_count >= MIN_DIVERSE_CELLS).astype(int)
        total_bad += bad_count

    idx = max(pool, key=lambda i: (n_tests_involved[i], total_bad[i]))
    print(
        f"[{product_name}] {tag} pick: {times[idx]} "
        f"({n_tests_involved[idx]} distinct tests involved, {total_bad[idx]} total bad cells)"
    )
    if require_multiple and n_tests_involved[idx] < 2:
        return None
    return times[idx]


def pick_flag_showcase(ds, product_name, test, tests, n_examples=1):
    """Timestep(s) where `test` fails the most cells *exclusively* -- i.e. no
    other individual test in `tests` is also bad on that same cell. Ranking
    on raw %-bad instead (the first version of this function) can pick a
    timestep where the target test's failures are entirely swamped by a
    co-occurring test (esp. the whole-scan RDCT_QC), producing a plot that
    shows none of the target test's own color at all. Returns a list of up to `n_examples`
    times (best first, each >= MIN_GAP_DAYS apart), empty if the test never
    fires exclusively at all."""
    if test in FLAGS_NEVER_FIRE:
        print(f"[{product_name}] {test}: skipped, never fires in this archive (WP2).")
        return []
    times = ds["time"].values
    substantial = substantial_coverage_mask(ds)
    pool = np.where(substantial)[0]

    target_bad = ds[test].squeeze("depth").isin(list(BAD_FLAGS))
    other_bad_any = None
    for t in tests:
        if t == test:
            continue
        ob = ds[t].squeeze("depth").isin(list(BAD_FLAGS))
        other_bad_any = ob if other_bad_any is None else (other_bad_any | ob)
    exclusive = target_bad if other_bad_any is None else (target_bad & ~other_bad_any)
    dims = [d for d in exclusive.dims if d != "time"]
    exclusive_counts = exclusive.sum(dim=dims).values

    if exclusive_counts[pool].max() == 0:
        print(
            f"[{product_name}] {test}: never fires without another test also failing on the same "
            f"cell within substantial-coverage timesteps -- no clean single-flag example exists, skipped."
        )
        return []
    picks = _pick_ranked(pool, exclusive_counts, times, n_examples)
    for rank, idx in enumerate(picks, start=1):
        tag = f"{test} showcase pick" if rank == 1 else f"{test} showcase pick #{rank}"
        print(f"[{product_name}] {tag}: {times[idx]} ({exclusive_counts[idx]} cells failing {test} exclusively)")
    return [times[i] for i in picks]


# Products where a real structural gap (cells with no EWCT data at all, not
# just QC-bad) is a meaningful thing to showcase, and how many examples --
# like WP2's total1/total2 screenshots, which show the GDOP-seam's underlying no-data gap. Only Total
# has this concept here -- a radial station's own coverage gaps are just its
# known outages, already covered by the good/bad and outage-window picks.
NO_DATA_GAP_PRODUCTS = {"Total": 2}
MIN_TYPICAL_COVERAGE = 0.5  # a cell counts as "normally populated" if real data exists here at least this fraction of the whole archive


def pick_no_data_gap(ds, product_name, n_examples=1):
    """Timestep(s) with the most 'normally populated' cells unexpectedly
    missing (raw EWCT NaN, not QC-bad) -- surfaces the structural GDOP-seam
    gap directly, rather than just its QC-flagged edge. A cell only counts
    toward the gap if it has real data at least MIN_TYPICAL_COVERAGE of the
    whole archive, so a cell that's structurally never covered by any
    station's combined geometry doesn't count as a 'gap' just for being its
    usual self."""
    times = ds["time"].values
    substantial = substantial_coverage_mask(ds)
    pool = np.where(substantial)[0]

    ewct = ds["EWCT"].squeeze("depth").values  # (time, lat, lon)
    is_nan = np.isnan(ewct)
    typical_coverage = (~is_nan).mean(axis=0)
    usually_populated = typical_coverage >= MIN_TYPICAL_COVERAGE
    gap_count = (is_nan & usually_populated[np.newaxis, :, :]).sum(axis=(1, 2))

    picks = _pick_ranked(pool, gap_count, times, n_examples)
    for rank, idx in enumerate(picks, start=1):
        tag = "no-data gap pick" if rank == 1 else f"no-data gap pick #{rank}"
        print(f"[{product_name}] {tag}: {times[idx]} ({gap_count[idx]} normally-populated cells missing)")
    return [times[i] for i in picks]


def compute_flag_grid(snap, tests):
    """Per-cell integer code: 0 = every individual test passes (uncolored),
    1..len(tests) = that one test failed, len(tests)+1 = 2+ tests failed at
    once. Cells with no QC evaluation at all this timestep (QCflag NaN) are
    masked out entirely, so only cells with real coverage draw anything."""
    bad_stack = np.stack([snap[test].isin(list(BAD_FLAGS)).values for test in tests], axis=0)
    n_bad = bad_stack.sum(axis=0)
    single_test_idx = np.argmax(bad_stack, axis=0) + 1  # only meaningful where n_bad == 1
    code = np.where(n_bad == 1, single_test_idx, 0)
    code = np.where(n_bad >= 2, len(tests) + 1, code)
    no_data = np.isnan(snap["QCflag"].values)
    return np.ma.masked_where(no_data, code)


MAX_COMBOS_IN_LEGEND = 3


def multi_flag_label(snap, tests):
    """Legend text for the 'multiple flags' color, naming which specific
    tests actually co-occur in this snapshot's multi-flag cells (most
    common combination first), so the legend says what the gray is made of."""
    bad_stack = np.stack([snap[t].isin(list(BAD_FLAGS)).values for t in tests], axis=0)
    n_bad = bad_stack.sum(axis=0)
    combo_counts = {}
    for i, j in np.argwhere(n_bad >= 2):
        combo = tuple(t for k, t in enumerate(tests) if bad_stack[k, i, j])
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    if not combo_counts:
        return "multiple flags"
    ranked = sorted(combo_counts.items(), key=lambda kv: -kv[1])
    shown = [" + ".join(c) for c, _ in ranked[:MAX_COMBOS_IN_LEGEND]]
    label = "multiple flags: " + "; ".join(shown)
    if len(ranked) > MAX_COMBOS_IN_LEGEND:
        label += f"; +{len(ranked) - MAX_COMBOS_IN_LEGEND} more combo(s)"
    return label


def _cell_edges(centers):
    """Cell edges for pcolormesh from 1-D regularly-spaced grid centers."""
    step = np.diff(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = centers[:-1] + step / 2
    edges[0] = centers[0] - step[0] / 2
    edges[-1] = centers[-1] + step[-1] / 2
    return edges


def plot_qc_flagged_field(ds, time, product_name, tests, label, out_dir, title_label=None):
    snap = ds.sel(time=time).squeeze("depth")
    u_raw = snap["EWCT"].values
    v_raw = snap["NSCT"].values
    speed = np.ma.masked_invalid(np.sqrt(u_raw**2 + v_raw**2))
    speed_ms = speed  # netCDF EWCT/NSCT are already m/s; kept in m/s (not cm/s) so the colorbar reads directly against the QC tests' m/s thresholds (e.g. CSPD_QC)
    # Arrows normalized to unit length -- direction only. Magnitude is
    # already shown by the colorbar, so varying arrow length too was
    # redundant and made dense/high-speed scans harder to read. Real zero-speed cells get a (0,0) vector, which
    # quiver draws as a point, same as it always did.
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(speed.filled(0) > 0, u_raw / speed.filled(1), 0.0)
        v = np.where(speed.filled(0) > 0, v_raw / speed.filled(1), 0.0)
    u = np.ma.masked_where(np.isnan(u_raw), u)
    v = np.ma.masked_where(np.isnan(v_raw), v)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    code = compute_flag_grid(snap, tests)

    colors = (
        [(0.0, 0.0, 0.0, 0.0)]
        + [to_rgba(TEST_COLORS[t], alpha=PATCH_ALPHA) for t in tests]
        + [to_rgba(MULTI_COLOR, alpha=PATCH_ALPHA)]
    )
    cmap = ListedColormap(colors)
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    bounds = np.arange(-0.5, len(colors) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    pad_lon, pad_lat = 0.02, 0.02
    extent = [lon.min() - pad_lon, lon.max() + pad_lon, lat.min() - pad_lat, lat.max() + pad_lat]
    scale = QUIVER_SCALE[product_name]
    ts_str = np.datetime_as_string(time, unit="m")

    fig = plt.figure(figsize=(8.2, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        facecolor="#d8d4c8", edgecolor="#5a5a5a", linewidth=0.6, zorder=2,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#c9c7bf", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    # Cartopy's GeoAxes + gridliner doesn't honor plain set_xlabel/set_ylabel
    # (the text is computed but never actually drawn) -- placed explicitly in
    # axes-fraction coordinates instead, offset past the gridliner's own
    # degree tick labels.
    ax.text(0.5, -0.09, "Longitude [°E]", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.text(-0.14, 0.5, "Latitude [°N]", transform=ax.transAxes, ha="center", va="center", rotation="vertical", fontsize=9)

    ax.pcolormesh(
        _cell_edges(lon), _cell_edges(lat), code,
        transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
        edgecolors="#c9c7bf", linewidth=0.15, zorder=2.5,
    )

    # Speed-colored quiver (the alternative style already used/compared in
    # WP2) instead of a single plain color: with a QC-flag color already
    # underneath, long plain-navy arrows were hard to tell apart from the flag patches.
    # Its colorbar is a second, separate legend from the categorical QC-flag
    # one below the axes, so the two don't compete for the same space.
    q = ax.quiver(
        lon2d, lat2d, u, v, speed_ms,
        transform=ccrs.PlateCarree(), cmap="viridis",
        scale=scale, width=0.0035, zorder=3,
    )
    cb = fig.colorbar(q, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("Speed [m/s]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    legend_handles = [Patch(facecolor=TEST_COLORS[t], alpha=PATCH_ALPHA, label=t) for t in tests]
    legend_handles.append(Patch(facecolor=MULTI_COLOR, alpha=PATCH_ALPHA, label=multi_flag_label(snap, tests)))
    leg = ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        fontsize=7, frameon=False, title="QC flag", title_fontsize=8, ncol=2,
    )
    # Left-align the "QC flag" title over the entries instead of centering it
    # -- centered, it read as an x-axis label sitting under the plot.
    leg._legend_box.align = "left"

    display_label = label if title_label is None else title_label
    ax.set_title(
        f"{product_name} surface current field + QC flags — {ts_str.replace('T', ' ')} UTC ({display_label})",
        fontsize=10, loc="left",
    )

    out_path = out_dir / f"qc_flagged_field_{product_name.lower()}_{label}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# Diverging, centered at 0: red = away from station (+), blue = toward (-).
# Matches the HFR Node's own NRT viewer (dark navy -> light blue -> pale
# yellow -> orange -> dark red); matplotlib's stock `RdYlBu_r` reproduces it
# almost exactly.
RDVA_CMAP = "RdYlBu_r"


def plot_rdva_field(ds, time, product_name, tests, label, out_dir, title_label=None):
    """Signed RDVA as a plain per-cell color field -- no arrows, no per-test
    color patches (see module docstring's "RDVA field reproductions" note
    for why). Radial products only; caller must not pass a product without
    an RDVA variable (Total)."""
    snap = ds.sel(time=time).squeeze("depth")
    rdva = np.ma.masked_invalid(snap["RDVA"].values)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    code = compute_flag_grid(snap, tests)
    bad = (~np.ma.getmaskarray(code)) & (code.filled(0) > 0)

    vmax = np.abs(rdva).max() if rdva.count() else 1.0
    pad_lon, pad_lat = 0.02, 0.02
    extent = [lon.min() - pad_lon, lon.max() + pad_lon, lat.min() - pad_lat, lat.max() + pad_lat]
    ts_str = np.datetime_as_string(time, unit="m")

    fig = plt.figure(figsize=(8.2, 6), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        facecolor="#d8d4c8", edgecolor="#5a5a5a", linewidth=0.6, zorder=2,
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#c9c7bf", linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.text(0.5, -0.09, "Longitude [°E]", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    ax.text(-0.14, 0.5, "Latitude [°N]", transform=ax.transAxes, ha="center", va="center", rotation="vertical", fontsize=9)

    mesh = ax.pcolormesh(
        _cell_edges(lon), _cell_edges(lat), rdva,
        transform=ccrs.PlateCarree(), cmap=RDVA_CMAP, vmin=-vmax, vmax=vmax,
        edgecolors="#c9c7bf", linewidth=0.15, zorder=2.5,
    )
    cb = fig.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("RDVA [m/s]  (+ away from station, − toward)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Bright-pink hatch for QC-bad cells, drawn with ax.pcolor (not
    # pcolormesh): pcolor omits a masked cell's polygon entirely rather than
    # drawing it transparent, so only bad cells get a hatched polygon at all
    # -- a single-entry "none" colormap keeps the fill itself invisible,
    # leaving just the hatch lines (colored via edgecolor) on top of the
    # RDVA mesh. Deep pink (`#ff1493`) doesn't appear anywhere in RdYlBu_r,
    # so it can't be confused with real RDVA color at a glance.
    BAD_EDGE_COLOR = "#ff1493"
    bad_only = np.ma.masked_where(~bad, np.zeros(bad.shape))
    if bad_only.count():
        ax.pcolor(
            _cell_edges(lon), _cell_edges(lat), bad_only,
            transform=ccrs.PlateCarree(), cmap=ListedColormap(["none"]),
            hatch="////", edgecolor=BAD_EDGE_COLOR, linewidth=0.0, zorder=3,
        )

    legend_handles = [Patch(
        facecolor="none", edgecolor=BAD_EDGE_COLOR, hatch="////",
        label="QC-bad (any individual test)",
    )]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        fontsize=7, frameon=False,
    )

    display_label = label if title_label is None else title_label
    ax.set_title(
        f"{product_name} RDVA (radial velocity) + QC flags — {ts_str.replace('T', ' ')} UTC ({display_label})",
        fontsize=10, loc="left",
    )

    out_path = out_dir / f"qc_rdva_field_{product_name.lower()}_{label}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for product_name, (path, tests) in PRODUCTS.items():
        print(f"\n=== [{product_name}] opening {path.name} ===")
        ds = xr.open_dataset(path)

        print(f"-- [{product_name}] good/bad --")
        good_time, bad_time = pick_good_bad(ds, product_name)
        # title_label spells out what "good"/"bad" actually means (the overall
        # QCflag), since "good"/"bad" alone read as ambiguous. Filenames (the `label` arg) stay good/bad, unchanged.
        plot_qc_flagged_field(ds, good_time, product_name, tests, "good", FIGURES_DIR, title_label="QCflag=good")
        plot_qc_flagged_field(ds, bad_time, product_name, tests, "bad", FIGURES_DIR, title_label="QCflag=bad")

        print(f"-- [{product_name}] diverse --")
        diverse_time = pick_diverse(ds, product_name, tests)
        if diverse_time is not None:
            plot_qc_flagged_field(ds, diverse_time, product_name, tests, "diverse", FIGURES_DIR)

        print(f"-- [{product_name}] per-flag showcase --")
        for test in tests:
            n = N_SHOWCASE_EXAMPLES.get(test, 1)
            picks = pick_flag_showcase(ds, product_name, test, tests, n_examples=n)
            for rank, t in enumerate(picks, start=1):
                suffix = f"flag-{test}" if rank == 1 else f"flag-{test}-{rank}"
                plot_qc_flagged_field(ds, t, product_name, tests, suffix, FIGURES_DIR)

        if product_name in NO_DATA_GAP_PRODUCTS:
            print(f"-- [{product_name}] no-data gap --")
            gap_picks = pick_no_data_gap(ds, product_name, n_examples=NO_DATA_GAP_PRODUCTS[product_name])
            for rank, t in enumerate(gap_picks, start=1):
                suffix = "no-data-gap" if rank == 1 else f"no-data-gap-{rank}"
                plot_qc_flagged_field(ds, t, product_name, tests, suffix, FIGURES_DIR)

        if product_name in OUTAGE_WINDOWS:
            print(f"-- [{product_name}] outage window --")
            outage_time = pick_diverse(
                ds, product_name, tests, time_window=OUTAGE_WINDOWS[product_name],
                require_multiple=False, tag="outage",
            )
            if outage_time is not None:
                plot_qc_flagged_field(ds, outage_time, product_name, tests, "outage", FIGURES_DIR)
                # RDVA version of the same pick: `pick_diverse` is deterministic over unchanged data, so
                # reusing `outage_time` here reproduces that exact timestamp
                # rather than re-picking independently.
                if product_name in STATIONS:
                    plot_rdva_field(ds, outage_time, product_name, tests, "outage", FIGURES_DIR)

        ds.close()

    print("\n=== Known-example reproductions (from the WP2 report screenshots) ===")
    for product_name, timestamp, label in KNOWN_EXAMPLES:
        path, tests = PRODUCTS[product_name]
        print(f"\n[{product_name}] {label} @ {timestamp}")
        ds = xr.open_dataset(path)
        plot_qc_flagged_field(ds, np.datetime64(timestamp), product_name, tests, label, FIGURES_DIR)
        ds.close()

    print("\n=== RDVA field reproductions (screenshot examples, radial products only) ===")
    for product_name, timestamp, label in KNOWN_EXAMPLES:
        if product_name not in STATIONS:
            continue  # Total has no RDVA/DRVA
        path, tests = PRODUCTS[product_name]
        print(f"\n[{product_name}] {label} @ {timestamp}")
        ds = xr.open_dataset(path)
        plot_rdva_field(ds, np.datetime64(timestamp), product_name, tests, label, FIGURES_DIR)
        ds.close()

    print("\n=== Named bora-storm examples ===")
    for product_name, timestamp, label, title_label in BORA_EXAMPLES:
        path, tests = PRODUCTS[product_name]
        print(f"\n[{product_name}] {label} @ {timestamp}")
        ds = xr.open_dataset(path)
        plot_qc_flagged_field(ds, np.datetime64(timestamp), product_name, tests, label, FIGURES_DIR, title_label=title_label)
        ds.close()


if __name__ == "__main__":
    main()
