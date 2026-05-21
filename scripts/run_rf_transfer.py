#!/usr/bin/env python3
"""Evaluate MLP-to-RF transfer attacks (gradient-free on RF)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config
from src.rf_transfer import run_rf_transfer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RF on MLP-crafted adversarial examples (transfer attack)"
    )
    parser.add_argument(
        "--baseline-run",
        default=None,
        help="Baseline run id under results/baselines/ (default: latest or config)",
    )
    parser.add_argument(
        "--attack",
        choices=("fgsm", "pgd"),
        default="fgsm",
        help="MLP white-box attack when not loading x_adv",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="L_inf radius (default: first value in config attacks.epsilon_values)",
    )
    parser.add_argument(
        "--x-adv-path",
        type=Path,
        default=None,
        help="Load precomputed adversarial examples (.npy or .npz with x_adv)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Cap test samples for a fast transfer smoke run",
    )
    parser.add_argument(
        "--no-save-x-adv",
        action="store_true",
        help="Skip saving generated x_adv arrays",
    )
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    run_rf_transfer(
        cfg,
        baseline_run=args.baseline_run,
        attack=args.attack,
        epsilon=args.epsilon,
        x_adv_path=args.x_adv_path,
        pilot=args.pilot,
        save_x_adv=not args.no_save_x_adv,
    )


if __name__ == "__main__":
    main()
