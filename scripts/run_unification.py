"""Unify all downloaded HFR-NAdr datasets (2021-present) into a consistent
format for Sklop 2 analyses: regularized time axis + harmonized attributes.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from download import DATASETS  # noqa: E402
from unify import unify_dataset  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    for name, cfg in DATASETS.items():
        if name == "total":
            raw_path = RAW_DIR / "total" / "hfr_nadr_total_2021_present.nc"
            out_path = PROCESSED_DIR / "total" / "hfr_nadr_total_2021_present_unified.nc"
        else:
            raw_path = RAW_DIR / "radials" / name / f"hfr_nadr_{name.lower()}_2021_present.nc"
            out_path = (
                PROCESSED_DIR / "radials" / name
                / f"hfr_nadr_{name.lower()}_2021_present_unified.nc"
            )

        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw file not found for {name}: {raw_path}\n"
                "Run scripts/run_full_download.py first."
            )

        print(f"Unifying {name}...")
        result = unify_dataset(raw_path, out_path, dataset_name=name, source_id=cfg["id"])
        print(f"  -> {result}")


if __name__ == "__main__":
    main()
