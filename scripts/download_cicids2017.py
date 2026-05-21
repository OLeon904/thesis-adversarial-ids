#!/usr/bin/env python3
"""Download CICIDS2017 MachineLearningCSV.zip from Hugging Face mirror."""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
REPO_ID = "bencorn/CICIDS2017"
ZIP_NAME = "csvs/MachineLearningCSV.zip"


def main() -> None:
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = RAW_DIR / ZIP_NAME

    print(f"Downloading {ZIP_NAME} from Hugging Face ({REPO_ID})...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=ZIP_NAME,
    )
    zip_path = Path(local_path)
    print(f"Downloaded to cache: {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB)")

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"{zip_path} is not a valid zip file")

    extract_dir = RAW_DIR / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    print("Extracting CSV files...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Flatten: zip may contain MachineLearningCSV/ subfolder
    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV files found in archive")

    for src in csv_files:
        dest = RAW_DIR / src.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))
        print(f"  -> {dest.name}")

    shutil.rmtree(extract_dir, ignore_errors=True)
    # Remove synthetic smoke file if present
    synth = RAW_DIR / "synthetic_smoke.csv"
    if synth.exists():
        synth.unlink()
        print("Removed synthetic_smoke.csv")

    print(f"Done. {len(list(RAW_DIR.glob('*.csv')))} CSV files in {RAW_DIR}")


if __name__ == "__main__":
    main()
