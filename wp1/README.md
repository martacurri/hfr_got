# WP1 — System inventory

[`wp1.pdf`](wp1.pdf) — *Collecting and editing metadata from HFR GOT* (Curri,
2026). System inventory report for the HFR-NAdr network: station metadata,
operating frequencies and resolution, measured variables, QC test
thresholds, and data availability (2015-2026 for currents; 2021-present
publicly available here).

## Pipeline code

| File | Description |
|---|---|
| `src/download.py` | Downloads raw station data from the EU HFR Node ERDDAP |
| `src/unify.py` | Regularizes and unifies the raw downloads into the published netCDF files |
| `scripts/run_full_download.py` | Entry point: runs the full download step |
| `scripts/run_unification.py` | Entry point: runs the unification step |

Requires the environment in [`../environment.yml`](../environment.yml):

```
conda env create -f ../environment.yml
conda activate hfr-qc
python scripts/run_full_download.py
python scripts/run_unification.py
```

Output data is published on the [data page](https://martacurri.github.io/hfr_got/data.html).
