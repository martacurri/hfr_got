"""WP3: validation, tuning and report for the kernel QC flag.

Validation set = all days of the test-set cases + 2025-12
(Bora winter, Christmas storm) + 2026-07 (calm summer, all four stations) +
2024-10 (most IZOL west-edge windows). The flag is run and reported on these
timestamps only.

Tuning: each parameter is varied on its own around the
defaults; a setting is better if it passes more test-set cases, or as many with
a lower share of 3/4 cells in the three validation months.

Outputs (outputs/reports/): kernel_qc_tuning.csv, kernel_qc_params.json,
kernel_qc_testset.csv, kernel_qc_summary.csv, kernel_qc_windows_<product>.csv,
kernel_qc_total_sources.csv; figures kernel_qc_field_<product>_<time>.png.

Run: python scripts/kernel_qc_report.py [--tune] [--report]
"""
import argparse
import dataclasses
import json

import numpy as np
import pandas as pd

from kernel_features import PROJECT_ROOT
from kernel_qc_flag import BAD, PROB_BAD, RADIALS, Params, load_inputs, run
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs

from kernel_plots import FIGURES_DIR, _draw_field, _draw_radial_vectors, _map_axes, _radial_uv
from plot_qc_flagged_field import _cell_edges

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PARAMS_PATH = REPORTS_DIR / "kernel_qc_params.json"
VALIDATION_MONTHS = ["2025-12", "2026-07", "2024-10"]
PRODUCTS = ["total"] + RADIALS


def _c(product, time, lat, lon, label, expect):
    return {"product": product, "time": time, "lat": lat, "lon": lon, "label": label, "expect": expect}


# Expert-labelled reference cases (WP2/WP3 examples): lat/lon = best 3x3 window centre or the named cell;
# None = check the whole field at that time (or day).
TEST_SET = [
    _c("total", "2023-12-31T04:30", 45.689, 13.510, "Total #1", "bad"),
    _c("total", "2024-10-22T08:00", 45.662, 13.472, "Total #2", "bad"),
    _c("total", "2024-02-02T08:00", 45.729, 13.568, "Total #3", "bad"),
    _c("total", "2023-11-11T10:00", 45.675, 13.549, "Total #5", "bad"),
    _c("total", "2023-12-28T01:00", 45.675, 13.529, "Total #6", "bad"),
    _c("total", "2023-02-02T20:00", 45.594, 13.568, "Total #8 (reference)", "bad"),
    _c("total", "2023-06-21T07:00", 45.635, 13.394, "Total #10", "bad"),
    _c("total", "2025-11-08T07:00", None, None, "Total WP2 divergence example", "bad"),
    _c("auri", "2022-03-29T11:30", 45.635, 13.607, "AURI shear", "bad"),
    _c("auri", "2026-02-25T01:00", 45.648, 13.626, "AURI shear", "bad"),
    _c("auri", "2022-01-02T11:00", 45.700, 13.600, "AURI WP2 diagonal", "bad"),
    _c("pira", "2022-02-23T17:00", 45.581, 13.626, "PIRA divergence", "bad"),
    _c("pira", "2023-04-23T05:00", 45.648, 13.587, "PIRA divergence", "bad"),
    _c("pira", "2024-08-08T01:30", 45.608, 13.529, "PIRA divergence", "bad"),
    _c("pira", "2025-07-09T20:30", 45.594, 13.607, "PIRA shear", "bad"),
    _c("pira", "2022-03-23T07:30", 45.608, 13.568, "PIRA shear", "bad"),
    _c("pira", "2024-12-05T10:30", None, None, "PIRA WP2 north patch", "bad"),
    _c("tri1", "2024-06-11T17:00", 45.621, 13.414, "TRI1 #1", "bad"),
    _c("tri1", "2024-10-22T03:30", 45.608, 13.684, "TRI1 #8 (reference)", "bad"),
    _c("tri1", "2024-03-15T08:30", 45.608, 13.684, "TRI1 #9 (reference)", "bad"),
    _c("tri1", "2024-03-03T19:30", 45.635, 13.645, "TRI1 #6 single-cell spike", "bad"),
    _c("izol", "2024-06-28T08:00", 45.594, 13.510, "IZOL #4 (reference)", "bad"),
    _c("izol", "2024-07-08T03:30", 45.668, 13.664, "IZOL #6 in-field patch", "bad"),
    _c("izol", "2023-06-28T22:00", 45.581, 13.665, "IZOL #7", "bad"),
    _c("izol", "2023-11-01T22:00", 45.648, 13.529, "IZOL #8", "bad"),
    _c("izol", "2024-04-08T23:00", 45.610, 13.530, "IZOL WP2 block", "bad"),
    _c("pira", "2022-06-24T10:30", None, None, "PIRA smooth zero-crossing", "not_bad"),
    _c("tri1", "2023-10-02T04:30", None, None, "TRI1 near-zero noise", "not_bad"),
    _c("tri1", "2023-11-03T04:00", 45.635, 13.529, "TRI1 #5 real front", "not_bad"),
    _c("total", "2024-10-01T17:00", 45.608, 13.645, "Total #7 Bora jet", "not_bad"),
    _c("total", "2025-12-25", None, None, "Christmas 2025 Bora (whole day)", "not_bad"),
]


def test_days():
    return sorted({c["time"][:10] for c in TEST_SET})


def validation_mask(times, months, days):
    t = np.asarray(times)
    return (np.isin(t.astype("datetime64[M]").astype(str), months)
            | np.isin(t.astype("datetime64[D]").astype(str), days))


def case_code(krev, times, lat_grid, lon_grid, case):
    """Worst KREV_QC code at a test case (0 if the time is not in `times`)."""
    tstr = np.asarray(times).astype("datetime64[m]").astype(str)
    hit = np.nonzero(np.char.startswith(tstr, case["time"]))[0]
    if len(hit) == 0:
        return 0
    k = krev[hit]
    if case["lat"] is not None:
        ci = int(np.abs(np.asarray(lat_grid) - case["lat"]).argmin())
        cj = int(np.abs(np.asarray(lon_grid) - case["lon"]).argmin())
        k = k[:, max(0, ci - 1):ci + 2, max(0, cj - 1):cj + 2]
    return int(k.max())


def evaluate(inp, p):
    days = test_days()
    runs, cases = {}, []
    for product in PRODUCTS:
        sel = validation_mask(inp.rows[product]["times"], VALIDATION_MONTHS, days)
        runs[product] = run(product, inp, sel, p)
    for c in TEST_SET:
        r = runs[c["product"]]
        f = inp.fields[c["product"]]
        code = case_code(r["cells"]["KREV_QC"], r["times"], f["lat"], f["lon"], c)
        flagged = code in (PROB_BAD, BAD)
        cases.append({**c, "passed_eu_qc": "yes", "code": code,
                      "ok": flagged if c["expect"] == "bad" else not flagged})
    bad = tested = 0
    for product, r in runs.items():
        in_months = validation_mask(r["times"], VALIDATION_MONTHS, [])
        k = r["cells"]["KREV_QC"][in_months]
        bad += int(np.isin(k, (PROB_BAD, BAD)).sum())
        tested += int((k > 0).sum())
    return runs, pd.DataFrame(cases), bad / max(tested, 1)


SWEEP = {
    "top_frac": [0.0025, 0.005, 0.01],
    "dv_min": [0.2, 0.3, 0.4],
    "s_min_total": [0.05, 0.10, 0.15],
    "s_min_radial": [0.03, 0.05, 0.08],
    "theta": [90.0, 120.0, 150.0],
    "jump": [0.2, 0.3, 0.4],
    "min_cells": [3, 5, 8],
    "pad": [0, 1, 2],
}


def _summary_row(p, testset, share):
    must, must_not = testset[testset.expect == "bad"], testset[testset.expect == "not_bad"]
    return {**dataclasses.asdict(p), "must_ok": int(must.ok.sum()), "must_n": len(must),
            "must_not_ok": int(must_not.ok.sum()), "must_not_n": len(must_not),
            "share_3_4_months": share,
            "failing": "; ".join(f"{r.label} ({r.time}) code {r.code}" for r in testset[~testset.ok].itertuples())}


def tune(inp):
    base = Params()
    seen, rows = set(), []
    for name, values in SWEEP.items():
        for v in values:
            p = dataclasses.replace(base, **{name: v})
            if p in seen:
                continue
            seen.add(p)
            _, testset, share = evaluate(inp, p)
            rows.append({"varied": name, **_summary_row(p, testset, share)})
            print(f"  {name}={v}: must {rows[-1]['must_ok']}/{rows[-1]['must_n']}, "
                  f"must-not {rows[-1]['must_not_ok']}/{rows[-1]['must_not_n']}, 3/4 share {share:.4%}")
    table = pd.DataFrame(rows)
    table["passes"] = table.must_ok + table.must_not_ok
    order = table.sort_values(["passes", "share_3_4_months"], ascending=[False, True])
    best_row = order.iloc[0]
    best = Params(**{f.name: type(getattr(base, f.name))(best_row[f.name]) for f in dataclasses.fields(Params)})
    return best, table


def load_params():
    if not PARAMS_PATH.exists():
        return Params()
    return Params(**json.loads(PARAMS_PATH.read_text()))


CODE_COLOURS = {PROB_BAD: "#ff7f0e", BAD: "#d62728"}  # 3 orange, 4 red; feature flag = blue outline
FEATURE_COLOUR = "#1f77b4"


def period_of(times):
    t = np.asarray(times).astype("datetime64[M]").astype(str)
    return np.where(np.isin(t, VALIDATION_MONTHS), t, "test days")


def summary_rows(product, run_out):
    rows = []
    per = period_of(run_out["times"])
    tper = period_of(run_out["table"]["time"].to_numpy()) if len(run_out["table"]) else np.array([], str)
    for period in [*VALIDATION_MONTHS, "test days"]:
        m = per == period
        if not m.any():
            continue
        k, kout = run_out["cells"]["KREV_QC"][m], run_out["cells"]["KOUT_FLAG"][m]
        tested = int((k > 0).sum())
        w = run_out["table"][tper == period]
        n = max(len(w), 1)
        row = {"product": product, "period": period, "timestamps": int(m.sum()), "tested_cells": tested,
               "pct_3": 100 * int((k == PROB_BAD).sum()) / max(tested, 1),
               "pct_4": 100 * int((k == BAD).sum()) / max(tested, 1),
               "pct_feature": 100 * int(((kout == 1) & (k > 0)).sum()) / max(tested, 1),
               "windows": len(w),
               "pct_a_only": 100 * int((w.picked_a & ~w.picked_b).sum()) / n,
               "pct_b_only": 100 * int((~w.picked_a & w.picked_b).sum()) / n,
               "pct_both_ab": 100 * int((w.picked_a & w.picked_b).sum()) / n,
               "pct_reversal": 100 * int((w.cls == "reversal").sum()) / n}
        for v, cnt in w.verdict.value_counts().items():
            row[f"verdict: {v}"] = cnt
        rows.append(row)
    return rows


def total_sources(table):
    rev = table[table.cls == "reversal"]
    west = rev[rev.win_lon < 13.45]
    return pd.DataFrame({"whole field": rev.verdict.value_counts(), "west edge (lon < 13.45)": west.verdict.value_counts()}
                        ).fillna(0).astype(int).rename_axis("verdict").reset_index()


def figure_path(product, stamp):
    """Output path of a flagged-hour figure; creates outputs/figures/ if missing."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"kernel_qc_field_{product}_{stamp.replace(':', '')}.png"


def plot_flagged_hour(product, inp, run_out, i):
    """Field at local time index i with flagged cells outlined (red 4, orange 3, blue feature)."""
    f = inp.fields[product]
    lat, lon = f["lat"], f["lon"]
    t = run_out["times"][i]
    t_arch = int(np.nonzero(f["time"] == t)[0][0])
    k, kout = run_out["cells"]["KREV_QC"][i], run_out["cells"]["KOUT_FLAG"][i]
    radial = product != "total"
    fig = plt.figure(figsize=(13 if radial else 7.5, 6.5))
    ax = _map_axes(fig, (1, 2, 1) if radial else (1, 1, 1), lat, lon)
    art, label = _draw_field(ax, f, product, t_arch)
    fig.colorbar(art, ax=ax, shrink=0.7, label=label)
    axes = [ax]
    if radial:
        ax2 = _map_axes(fig, (1, 2, 2), lat, lon, left_labels=False)
        u, v = _radial_uv(product, t)
        _draw_radial_vectors(ax2, lat, lon, u, v)
        axes.append(ax2)
    le, ae = _cell_edges(lon), _cell_edges(lat)
    for a in axes:
        for (r, c) in zip(*np.nonzero(kout == 1)):
            a.add_patch(Rectangle((le[c], ae[r]), le[c + 1] - le[c], ae[r + 1] - ae[r], fill=False,
                                  edgecolor=FEATURE_COLOUR, linewidth=1.0, transform=ccrs.PlateCarree(), zorder=4))
        for code, colour in CODE_COLOURS.items():
            for (r, c) in zip(*np.nonzero(k == code)):
                a.add_patch(Rectangle((le[c], ae[r]), le[c + 1] - le[c], ae[r + 1] - ae[r], fill=False,
                                      edgecolor=colour, linewidth=1.8, transform=ccrs.PlateCarree(), zorder=5))
    stamp = str(t)[:16]
    fig.suptitle(f"{product.upper()} {stamp} UTC — kernel QC flag: red = 4 bad, orange = 3 probably bad, "
                 f"blue = outburst / confirmed real", fontsize=10)
    out = figure_path(product, stamp)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def write_report(inp, p):
    runs, testset, share = evaluate(inp, p)
    testset.to_csv(REPORTS_DIR / "kernel_qc_testset.csv", index=False)
    rows = []
    for product, r in runs.items():
        r["table"].to_csv(REPORTS_DIR / f"kernel_qc_windows_{product}.csv", index=False)
        rows += summary_rows(product, r)
    summary = pd.DataFrame(rows)
    summary.insert(2, "thr_a_m_s", summary["product"].map({pr: runs[pr]["thr_a"] for pr in runs}))
    summary.insert(3, "thr_b_score", summary["product"].map({pr: runs[pr]["thr_b"] for pr in runs}))
    summary.to_csv(REPORTS_DIR / "kernel_qc_summary.csv", index=False)
    total_sources(runs["total"]["table"]).to_csv(REPORTS_DIR / "kernel_qc_total_sources.csv", index=False)
    print(f"Params: {p}\nTest set: {int(testset.ok.sum())}/{len(testset)} ok; 3/4 share in months {share:.4%}")
    for product, r in runs.items():
        per = period_of(r["times"])
        n_bad = np.isin(r["cells"]["KREV_QC"], (PROB_BAD, BAD)).sum(axis=(1, 2))
        for month in VALIDATION_MONTHS:
            m = np.nonzero(per == month)[0]
            if len(m) and n_bad[m].max() > 0:
                print("Saved:", plot_flagged_hour(product, inp, r, int(m[n_bad[m].argmax()])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inp = load_inputs()
    if args.tune:
        best, table = tune(inp)
        table.to_csv(REPORTS_DIR / "kernel_qc_tuning.csv", index=False)
        PARAMS_PATH.write_text(json.dumps(dataclasses.asdict(best), indent=2))
        print(f"Saved: kernel_qc_tuning.csv, kernel_qc_params.json -> {best}")
    if args.report:
        write_report(inp, load_params())


if __name__ == "__main__":
    main()
