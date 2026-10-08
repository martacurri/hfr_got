# WP2 — Effectiveness of existing QC algorithms

[`wp2.pdf`](wp2.pdf) — effectiveness analysis of the existing EU HFR Node QC
algorithms on the HFR-NAdr archive (2021-present currents): inventory of the
existing tests, quantitative flag analysis, and upgrade recommendations for
WP3's new QC development.

## Analysis scripts

| File | Description |
|---|---|
| `scripts/qc_flag_analysis.py` | Flag histograms, monthly bad-fraction time series, and per-cell spatial bad-fraction maps for every existing QC test |

(The development of new QC algorithms for the artefacts the existing QC
misses is in [WP3](../wp3/).)

## 1. Set up the environment

```
conda env create -f ../environment.yml
conda activate hfr-qc
```

## 2. Get the data

The script reads 5 netCDF files. They are **not** stored in this repo (too
large for git) — download them from the [data page](https://martacurri.github.io/hfr_got/data.html)
and place each file in the matching folder below (folder names are
case-sensitive and must match exactly, since the script locates everything
relative to its own path):

```
data/processed/radials/AURI/hfr_nadr_auri_2021_present_unified.nc
data/processed/radials/PIRA/hfr_nadr_pira_2021_present_unified.nc
data/processed/radials/TRI1/hfr_nadr_tri1_2021_present_unified.nc
data/processed/radials/IZOL/hfr_nadr_izol_2021_present_unified.nc
data/processed/total/hfr_nadr_total_2021_present_unified.nc
```

## 3. Run

```
python scripts/qc_flag_analysis.py
```

Outputs are written to `outputs/reports/` and `outputs/figures/` (created
automatically).
