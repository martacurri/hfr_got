"""Download, regularize, and concatenate HFR-NAdr ERDDAP data."""
import calendar
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


def month_ranges(start_date: str, end_date: str | None = None):
    """Yield (year, month, period_start_iso, period_end_iso) for each
    calendar month between start_date and end_date inclusive, clamped to
    start_date/end_date at the boundaries.

    Dates are ISO 8601 strings (e.g. "2021-01-01" or "2021-01-01T00:00:00Z").
    If end_date is None, uses the current UTC time.
    """
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    if end_date is None:
        end = datetime.now(timezone.utc)
    else:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        period_start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
        period_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        if period_start < start:
            period_start = start
        if period_end > end:
            period_end = end

        yield (
            year,
            month,
            period_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            period_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


DEFAULT_BASE_URL = "https://erddap.hfrnode.eu/erddap/griddap/"


def build_erddap_url(
    dataset_id: str,
    variables: list[str],
    time_start: str,
    time_end: str,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Build an ERDDAP griddap .nc request URL.

    Dimension order is (time, depth, latitude, longitude). depth is fixed
    to its single index (0); latitude/longitude use "last" to request the
    full spatial extent regardless of the dataset's actual grid bounds.
    """
    constraint = f"[({time_start}):1:({time_end})][0:1:0][0:1:last][0:1:last]"
    var_clauses = ",".join(f"{v}{constraint}" for v in variables)
    return f"{base_url}{dataset_id}.nc?{var_clauses}"


def download_monthly_chunk(
    dataset_id: str,
    variables: list[str],
    year: int,
    month: int,
    time_start: str,
    time_end: str,
    output_dir,
    base_url: str = DEFAULT_BASE_URL,
    max_retries: int = 3,
    retry_backoff: float = 5.0,
):
    """Download one calendar month of data for dataset_id, returning the
    output file path. Returns the existing path without a network call if
    the file already exists. Returns None if ERDDAP reports no data for
    this period (HTTP 404). Raises RuntimeError if all retries fail for any
    other error.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_id}_{year}{month:02d}.nc"

    if output_path.exists():
        return output_path

    url = build_erddap_url(dataset_id, variables, time_start, time_end, base_url)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            return output_path
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            last_error = exc
        except requests.RequestException as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(retry_backoff * attempt)

    raise RuntimeError(
        f"Failed to download {dataset_id} {year}-{month:02d} after "
        f"{max_retries} attempts: {last_error}"
    )


def regularize_time_axis(ds: xr.Dataset, freq: str = "30min") -> xr.Dataset:
    """Reindex ds onto a regular time axis at the given frequency, spanning
    the dataset's existing min/max time. Missing timesteps become NaN.
    """
    full_index = pd.date_range(
        start=pd.Timestamp(ds.time.values.min()),
        end=pd.Timestamp(ds.time.values.max()),
        freq=freq,
    )
    return ds.reindex(time=full_index)


def concatenate_monthly_files(input_dir, dataset_id: str, output_path):
    """Concatenate all `<dataset_id>_YYYYMM.nc` files in input_dir along time
    and write the result to output_path. Raises FileNotFoundError if no
    matching files exist.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(input_dir.glob(f"{dataset_id}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No files found for {dataset_id} in {input_dir}")

    ds = xr.open_mfdataset(files, combine="by_coords")
    ds.load()

    # ERDDAP can snap a month's upper time bound into the next month when
    # there's a data gap right at the boundary, so adjacent monthly chunks
    # can share one identical timestamp.
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=np.sort(unique_idx))

    ds.to_netcdf(output_path)
    ds.close()
    return output_path


def download_dataset(
    dataset_id: str,
    variables: list[str],
    time_start: str,
    output_dir,
    combined_output_path,
    time_end: str | None = None,
    regularize: bool = False,
    base_url: str = DEFAULT_BASE_URL,
):
    """Download all monthly chunks for dataset_id from time_start to
    time_end (default: now), concatenate them, optionally regularize the
    time axis (Total dataset only), and write the combined netCDF.
    """
    output_dir = Path(output_dir)
    combined_output_path = Path(combined_output_path)

    for year, month, period_start, period_end in month_ranges(time_start, time_end):
        download_monthly_chunk(
            dataset_id,
            variables,
            year,
            month,
            period_start,
            period_end,
            output_dir,
            base_url=base_url,
        )

    concatenate_monthly_files(output_dir, dataset_id, combined_output_path)

    if regularize:
        ds = xr.open_dataset(combined_output_path)
        regularized = regularize_time_axis(ds)
        ds.close()
        regularized.to_netcdf(combined_output_path)
        regularized.close()

    return combined_output_path


RADIAL_VARIABLES = [
    "RDVA", "EWCT", "NSCT", "DRVA", "HCSS", "EACC",
    "QCflag", "OWTR_QC", "CSPD_QC", "VART_QC", "MDFL_QC", "AVRB_QC",
    "RDCT_QC", "POSITION_QC",
]

TOTAL_VARIABLES = [
    "EWCT", "NSCT", "UACC", "VACC", "GDOP",
    "QCflag", "VART_QC", "CSPD_QC", "DDNS_QC", "GDOP_QC", "POSITION_QC",
]

DATASETS = {
    "total": {
        "id": "EUHFR_NRTcurrent_HFR-NAdr-Total_v3",
        "variables": TOTAL_VARIABLES,
        "regularize": True,
    },
    "AURI": {
        "id": "EUHFR_NRTcurrent_HFR-NAdr-AURI_v3",
        "variables": RADIAL_VARIABLES,
        "regularize": False,
    },
    "IZOL": {
        "id": "EUHFR_NRTcurrent_HFR-NAdr-IZOL_v3",
        "variables": RADIAL_VARIABLES,
        "regularize": False,
    },
    "PIRA": {
        "id": "EUHFR_NRTcurrent_HFR-NAdr-PIRA_v3",
        "variables": RADIAL_VARIABLES,
        "regularize": False,
    },
    "TRI1": {
        "id": "EUHFR_NRTcurrent_HFR-NAdr-TRI1_v3",
        "variables": RADIAL_VARIABLES,
        "regularize": False,
    },
}
