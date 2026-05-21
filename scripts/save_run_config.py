#!/usr/bin/env python3
"""CLI: copy the active config YAML into a run results directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import save_config_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Save config YAML snapshot for a run directory.")
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Run output directory (e.g. results/attacks/<run_id>)",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path (default: default.yaml)")
    parser.add_argument(
        "--dest-name",
        type=str,
        default="config_snapshot.yaml",
        help="Snapshot filename inside out_dir",
    )
    args = parser.parse_args()
    dest = save_config_snapshot(args.out_dir, args.config, dest_name=args.dest_name)
    print(dest)


if __name__ == "__main__":
    main()
