#!/usr/bin/env python3
"""Verify constrained projection differs in raw vs scaled space."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constraints import build_index_masks, classify_columns, project_batch
from src.constraints_torch import make_constraint_fn, project_batch_torch
from src.io_utils import load_processed_data, load_scaler_label_encoder
from src.config import load_config


def main() -> int:
    cfg = load_config()
    processed = Path(cfg["paths"]["processed_dir"])
    splits, meta = load_processed_data(processed)
    scaler, _ = load_scaler_label_encoder(processed)
    names = meta["feature_names"]
    groups = meta.get("feature_groups") or classify_columns(names)

    x_orig = splits["X_test"][:4].copy()
    x_adv = x_orig + 0.05

    # Scaled-space projection (legacy)
    scaled_only = project_batch_torch(
        __import__("torch").from_numpy(x_adv).float(),
        __import__("torch").from_numpy(x_orig).float(),
        groups,
        feature_names=names,
    ).numpy()

    # Raw-space projection
    fn = make_constraint_fn(names, groups, cfg, scaler=scaler)
    raw_path = fn(
        __import__("torch").from_numpy(x_adv).float(),
        __import__("torch").from_numpy(x_orig).float(),
    ).numpy()

    diff = np.abs(raw_path - scaled_only).max()
    print(f"max |raw_projection - scaled_projection| = {diff:.6f}")
    if diff < 1e-6:
        print("FAIL: projections identical — raw-space path may be broken")
        return 1

    # Integer feature should be near-integer in raw after raw-path projection
    _, discrete_idx, _, _, _ = build_index_masks(names, groups)
    if discrete_idx.size > 0:
        raw_adv = scaler.inverse_transform(raw_path)
        frac = np.abs(raw_adv[:, discrete_idx] - np.rint(raw_adv[:, discrete_idx])).max()
        print(f"max fractional part on discrete features (raw) = {frac:.6f}")
        if frac > 1e-5:
            print("WARN: discrete features not integral after raw projection")

    print("PASS: raw-space constraint path is active and differs from scaled-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
