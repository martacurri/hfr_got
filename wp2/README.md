# WP2 — Effectiveness of existing QC algorithms

[`wp2.pdf`](wp2.pdf) — effectiveness analysis of the existing EU HFR Node QC
algorithms on the HFR-NAdr archive (2021-present currents): inventory of the
existing tests, quantitative flag analysis, known-artefact gap analysis, and
upgrade recommendations for WP3.

## Analysis scripts

| File | Description |
|---|---|
| `scripts/qc_flag_analysis.py` | Flag histograms, monthly bad-fraction time series, and per-cell spatial bad-fraction maps for every existing QC test |
| `scripts/qc_gap_analysis.py` | Applies lightweight versions of five known-gap tests (divergence, vorticity, neighbor-consistency, temporal spike, climatology) to the Total product and quantifies what the existing QC misses |

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

## 3. Run

```
python scripts/qc_flag_analysis.py
python scripts/qc_gap_analysis.py
```

`qc_gap_analysis.py` only needs the `total` file; `qc_flag_analysis.py` needs
all 5. Outputs are written to `outputs/reports/` and `outputs/figures/`
(created automatically).
