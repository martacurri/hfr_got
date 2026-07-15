"""Harmonize netCDF attributes across HFR-NAdr datasets for unified analysis."""
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

from download import regularize_time_axis

CONVENTIONS = "CF-1.8"

INSTITUTION = (
    "OGS, NIB, ARSO, ARPA FVG (HFR-NAdr network); "
    "unified by the HFR Trieste QC project"
)

DATASET_TITLES = {
    "total": "HFR-NAdr Total Surface Currents (Gulf of Trieste), 2021-present",
    "AURI": "HFR-NAdr Radial Currents — AURI station (Aurisina, IT), 2021-present",
    "IZOL": "HFR-NAdr Radial Currents — IZOL station (Izola, SI), 2021-present",
    "PIRA": "HFR-NAdr Radial Currents — PIRA station (Piran, SI), 2021-present",
    "TRI1": "HFR-NAdr Radial Currents — TRI1 station (Trieste, IT), 2021-present",
}

LONG_NAMES = {
    "EWCT": "Eastward surface current velocity",
    "NSCT": "Northward surface current velocity",
    "UACC": "U-component current accuracy (uncertainty)",
    "VACC": "V-component current accuracy (uncertainty)",
    "GDOP": "Geometric dilution of precision",
}

STANDARD_NAMES = {
    "EWCT": "eastward_sea_water_velocity",
    "NSCT": "northward_sea_water_velocity",
}


def harmonize_attributes(ds: xr.Dataset, dataset_name: str, source_id: str) -> xr.Dataset:
    """Set consistent global/variable attributes on ds in place, without
    clobbering existing meaningful variable-level values. Returns ds.
    """
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    ds.attrs["Conventions"] = CONVENTIONS
    ds.attrs["title"] = DATASET_TITLES[dataset_name]
    ds.attrs["institution"] = INSTITUTION
    ds.attrs["source"] = (
        f"ERDDAP dataset {source_id}, "
        f"https://erddap.hfrnode.eu/erddap/griddap/{source_id}"
    )
    existing_history = ds.attrs.get("history", "")
    new_entry = (
        f"{download_date}: unified via hfr_trieste_qc src/unify.py "
        f"(source ERDDAP dataset {source_id})"
    )
    ds.attrs["history"] = f"{existing_history}\n{new_entry}".strip()

    for var, long_name in LONG_NAMES.items():
        if var in ds.data_vars:
            ds[var].attrs.setdefault("long_name", long_name)
    for var, standard_name in STANDARD_NAMES.items():
        if var in ds.data_vars:
            ds[var].attrs.setdefault("standard_name", standard_name)

    return ds


def unify_dataset(input_path, output_path, dataset_name: str, source_id: str, freq: str = "30min"):
    """Load input_path, regularize its time axis and harmonize attributes,
    then write the result to output_path. Returns output_path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_path)
    ds.load()
    regularized = regularize_time_axis(ds, freq=freq)
    ds.close()

    harmonize_attributes(regularized, dataset_name, source_id)
    encoding = {var: {"zlib": True, "complevel": 4} for var in regularized.data_vars}
    regularized.to_netcdf(output_path, encoding=encoding)
    regularized.close()
    return output_path
