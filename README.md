# HFR-NAdr — Gulf of Trieste

This repository holds the work on high-frequency radar (HFR) data for
**HFR-NAdr**, the network operating in the Gulf of Trieste (northern
Adriatic Sea). The network has four stations run by three institutions —
AURI (OGS), TRI1 (ARPA FVG), IZOL (NIB) and PIRA (ARSO) — measuring surface
current velocities since 2015, with data from 2021 onward published here.

Work is organized into work packages (WP), each with its own report and
code:

| | |
|---|---|
| **[WP1](wp1/)** | System inventory: station metadata, operating parameters, measured variables, QC test thresholds, and data availability across the network — plus the pipeline that downloads and unifies the published data. |
| **[WP2](wp2/)** | Effectiveness analysis of the existing EU HFR Node QC algorithms — inventories the current QC tests and quantifies what they catch across the archive — plus the analysis scripts. |
| **[WP3](wp3/)** | New QC algorithms for anomalies the existing QC misses — k-means regime clustering, sliding-window kernels with an Isolation Forest anomaly score, and a new per-cell QC flag for non-physical reversals (divergence, convergence, shear) with a cross-station check — plus the scripts. |

## Station map

Interactive map of the four HFR-NAdr stations, with institution, location,
frequency and manufacturer shown on hover.

**[-> Open the station map](https://martacurri.github.io/hfr_got/)**

## Reports

- **[`wp1/wp1.pdf`](wp1/wp1.pdf)** — *Collecting and editing metadata from HFR GOT* (Curri, 2026)
- **[`wp2/wp2.pdf`](wp2/wp2.pdf)** — *Evaluation of the effectiveness of existing QC algorithms* (Curri, 2026)
- **[`wp3/wp3.pdf`](wp3/wp3.pdf)** — *Development of new QC algorithms for the detection of anomalies that remain undetected by existing algorithms* (Curri, 2026)

## Data downloads

Unified current data (2021-present) for the Total field and each of the
four radial stations, as NetCDF files with a regularized 30-minute time
axis and CF-1.8-style attributes, produced by the WP1 pipeline from the
[EU HFR Node ERDDAP](https://erddap.hfrnode.eu/erddap/).

**[-> Open the data page](https://martacurri.github.io/hfr_got/data.html)**

The code of all work packages shares one Python environment — see
[`environment.yml`](environment.yml).
