"""Download the full HFR-NAdr archive (2021-present) for the Total dataset
and all 4 radial datasets from ERDDAP.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from download import DATASETS, download_dataset  # noqa: E402

START_DATE = "2021-01-01"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    for name, cfg in DATASETS.items():
        if name == "total":
            out_dir = RAW_DIR / "total" / "monthly"
            combined = RAW_DIR / "total" / "hfr_nadr_total_2021_present.nc"
        else:
            out_dir = RAW_DIR / "radials" / name / "monthly"
            combined = RAW_DIR / "radials" / name / f"hfr_nadr_{name.lower()}_2021_present.nc"

        print(f"Downloading {name} ({cfg['id']})...")
        result = download_dataset(
            dataset_id=cfg["id"],
            variables=cfg["variables"],
            time_start=START_DATE,
            output_dir=out_dir,
            combined_output_path=combined,
            regularize=cfg["regularize"],
        )
        print(f"  -> {result}")


if __name__ == "__main__":
    main()
