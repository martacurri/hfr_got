# HF Radar Network — Gulf of Trieste (HFR-NAdr)

This repository publishes station information and current-velocity data for
**HFR-NAdr**, the high-frequency radar (HFR) network operating in the Gulf
of Trieste (northern Adriatic Sea), across four stations run by three
institutions: AURI (OGS), TRI1 (ARPA), IZOL (NIB) and PIRA (ARSO).

**Contents of this repository:**

| File / folder | Description |
|---|---|
| `index.qmd`, `data.qmd` | Source pages for the published website (station map, data downloads) |
| `wp1.pdf` | Work Project 1 system inventory report |
| `got_map.png` | Station map image (GEBCO bathymetry base) |
| `src/download.py`, `src/unify.py` | Pipeline to download and unify station data from the EU HFR Node |
| `scripts/run_full_download.py`, `scripts/run_unification.py` | Entry-point scripts for the pipeline above |
| `environment.yml` | Python environment required to run the pipeline |
| `_quarto.yml`, `_site/` | Quarto website project/build files |

## Work Project 1 report

[`wp1.pdf`](wp1.pdf) — *Collecting and editing metadata from HFR GOT*
(Curri, 2026). This is the system inventory report for the network: station
metadata, operating frequencies and resolution, measured variables, QC test
thresholds, and data availability for HFR-NAdr (2015-2026 for currents;
2021-present publicly available here).

## Station map

An interactive map of the four HFR-NAdr stations, with institution,
location, frequency and manufacturer shown on hover.

**[-> Open the station map](https://martacurri.github.io/hfr_got/)**

## Data downloads

Unified current data (2021-present) for the Total field and each of the
four radial stations, as NetCDF files with a regularized 30-minute time
axis and CF-1.8-style attributes. `src/` and `scripts/` contain the
pipeline used to produce these files from the
[EU HFR Node ERDDAP](https://erddap.hfrnode.eu/erddap/) — see
`environment.yml` for the required Python environment.

**[-> Open the data page](https://martacurri.github.io/hfr_got/data.html)**
