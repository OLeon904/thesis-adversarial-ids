#!/usr/bin/env python3
"""Multi-pass adversarial training for the MLP (PGD-mixed batches)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adversarial_train import run_adversarial_training
from src.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial training for MLP (clean + PGD mix per batch)"
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Number of adversarial passes (default: config adversarial_training.passes)",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default=None,
        help="Baseline run id under results/baselines/ (default: latest)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Cap training rows for a fast pilot run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: config/default.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    run_adversarial_training(
        cfg,
        passes=args.passes,
        baseline_run=args.baseline_run,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
