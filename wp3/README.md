# WP3 — New QC algorithms for anomalies the existing QC misses

Development of new quality-control (QC) algorithms for the HFR-NAdr archive
(2021-present currents), aimed at non-physical divergences, convergences and
shear between neighbouring cells, which the existing EU HFR Node QC does not
detect (see [WP2](../wp2/)). The scripts are run on the same unified data as
WP1/WP2.

## Analysis scripts

| File | Description |
|---|---|
| `scripts/cluster_regimes.py` | k-means clustering (k = 2–8) of whole-domain snapshots, per product (Total and each radial station); cluster summaries and mean-field figures |
| `scripts/plot_cluster_panels.py` | all clusters of one Total k-means run in one figure (`K` = number of clusters) |
| `scripts/kernel_features.py` | sliding-window "convolution": four half-split kernels (top/bottom, left/right, two diagonals) on 4×4 and 3×3 windows of every timestamp |
| `scripts/kernel_clustering.py` | Isolation Forest anomaly score for every window |
| `scripts/kernel_plots.py` | kernel maps, anomaly maps and monthly rates, figures of the most anomalous events |
| `scripts/plot_radial_vectors.py` | a radial station's velocity field at one time (examples of artefacts the existing QC misses) |
| `scripts/kernel_qc_flag.py` | the new QC flag: reversal vs outburst, cross-station check, per-cell codes |
| `scripts/kernel_qc_report.py` | tuning, test set, validation months, summary tables and figures for the new QC flag |
| `scripts/plot_qc_flagged_field.py` | current fields with each cell's existing QC-test failures; also provides colormap/grid helpers used by the kernel figures |

### The new QC flag in short

1. **Find:** a complete 3×3 window (all 9 cells passed the existing QC) is
   flagged if its largest kernel value **or** its Isolation Forest score is in
   the product's top 0.5%.
2. **Sort:** *reversal* if two neighbouring cells point in opposite directions
   (angle > 90°; radials: opposite sign), both at least `s_min` (Total
   0.15 m/s, radials 0.05 m/s) and differing by at least 0.3 m/s; otherwise
   *outburst* (a strength jump in the same direction, e.g. a Bora jet edge).
3. **Cross-station check (radial reversals):** the other radial stations at the
   same time and place (window ± 1 cell). ≥ 2 stations also show a jump →
   probably a real front (not flagged); none shows a jump → **4 (bad)**;
   otherwise → **3 (probably bad)**.
4. **Total reversals:** always **4**; the radial station(s) showing a jump at
   that place are recorded as the likely source.

Every cell of a flagged window gets the window's result (worst wins).
Outbursts and confirmed fronts get a separate informational flag.

## 1. Set up the environment

```
conda env create -f ../environment.yml
conda activate hfr-qc
```

## 2. Get the data

The scripts read 5 netCDF files. They are **not** stored in this repo (too
large for git) — download them from the [data page](https://martacurri.github.io/hfr_got/data.html)
and place each file in the matching folder below (folder names are
case-sensitive and must match exactly, since the scripts locate everything
relative to their own path):

```
data/processed/radials/AURI/hfr_nadr_auri_2021_present_unified.nc
data/processed/radials/PIRA/hfr_nadr_pira_2021_present_unified.nc
data/processed/radials/TRI1/hfr_nadr_tri1_2021_present_unified.nc
data/processed/radials/IZOL/hfr_nadr_izol_2021_present_unified.nc
data/processed/total/hfr_nadr_total_2021_present_unified.nc
```

IZOL data before 2023-03-01 (test period before the station's official start)
is left out automatically.

## 3. Run

Run from this `wp3/` folder, in this order (each step uses the files written
by the previous ones). Times are for a laptop (Intel i7, 16 GB RAM).

**k-means clustering**

```
python scripts/cluster_regimes.py          # ~8 min
python scripts/plot_cluster_panels.py 2    # < 1 min; k = 2 figure, use 6 for k = 6
```

**Kernels, anomaly score and the new QC flag**

```
python scripts/kernel_features.py          # ~12 min: kernels for 4x4 and 3x3 windows
python scripts/kernel_clustering.py        # ~45 min: Isolation Forest scores
python scripts/kernel_qc_report.py --tune     # ~2 min: parameter tuning -> kernel_qc_params.json
python scripts/kernel_qc_report.py --report   # ~1.5 min: test set, validation months, figures
```

**Figures**

```
python scripts/kernel_plots.py                         # all kernel figures
python scripts/kernel_plots.py --top                   # ~11 min: only the most anomalous events
python scripts/kernel_plots.py --featuremap total 2024-10-22T08:00   # one product and time
python scripts/plot_radial_vectors.py izol 2024-04-08T23:00
python scripts/plot_qc_flagged_field.py                # ~5 min: existing-QC flags on the current field
```

`--report` uses the parameters saved by `--tune`; without them it uses the
defaults.

## Outputs

Written to `outputs/reports/` and `outputs/figures/` (created automatically);
the kernel values and anomaly scores go to `data/processed/kernels/`.

| File | Content |
|---|---|
| `outputs/reports/cluster_summary_<product>.csv` | k-means cluster sizes and silhouette scores per k |
| `outputs/reports/kernel_anomalies_top_<product>[_3x3].csv` | the top 1% most anomalous windows |
| `outputs/reports/kernel_qc_params.json` | the tuned QC-flag parameters |
| `outputs/reports/kernel_qc_tuning.csv` | every tried parameter setting and its result on the test set |
| `outputs/reports/kernel_qc_testset.csv` | reference cases: passed the existing QC, new flag code |
| `outputs/reports/kernel_qc_summary.csv` | per product and validation period: % of cells flagged 3 / 4 / outburst, thresholds |
| `outputs/reports/kernel_qc_windows_<product>.csv` | every flagged window: kernel values, reversal/outburst, cross-station result, code |
| `outputs/reports/kernel_qc_total_sources.csv` | Total reversals by the radial station(s) showing a jump |
| `outputs/figures/kernel_qc_field_<product>_<time>.png` | current field with flagged cells (red = 4, orange = 3, blue = outburst/confirmed) |
| `outputs/figures/cluster_*`, `kernel_*`, `qc_*` | the clustering, kernel and existing-QC figures |

The validation set is all days of the reference cases plus three months:
2025-12 (Bora winter), 2026-07 (calm summer, all four stations) and 2024-10
(most activity on IZOL's far-range west edge).
