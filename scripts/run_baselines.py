#!/usr/bin/env python3
"""Train Random Forest and MLP; report clean test metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config
from src.train_baselines import run_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    run_baselines(cfg, pilot=args.pilot)


if __name__ == "__main__":
    main()
