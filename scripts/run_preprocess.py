#!/usr/bin/env python3
"""Preprocess CICIDS2017 CSV files into train/val/test splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config
from src.preprocess import run_preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Cap rows via pilot_max_rows in config")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    run_preprocess(cfg, pilot=args.pilot)


if __name__ == "__main__":
    main()
