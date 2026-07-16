# HF Radar Network — Gulf of Trieste (HFR-NAdr)

Public data and station information for the HFR-NAdr high-frequency radar
network in the Gulf of Trieste (AURI, TRI1, IZOL, PIRA).

**[-> Visit the site](https://martacurri.github.io/hfr_got/)** for the
station map and downloadable current data (2021-present).

`src/` and `scripts/` contain the pipeline used to download and unify the
data from the [EU HFR Node ERDDAP](https://erddap.hfrnode.eu/erddap/) — see
`environment.yml` for the required Python environment.

## Updates

- **2026-07-16**: Added the WP1 system inventory report (`wp1.pdf`) —
  station metadata, operating frequencies/resolution, measured variables,
  QC test thresholds, and data availability for the HFR-NAdr network
  (2015-2026, currents; 2021-present publicly available). Updated the
  station map (`got_map.png`) with a refreshed GEBCO bathymetry
  visualization.
