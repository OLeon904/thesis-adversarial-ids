#!/usr/bin/env python3
"""
Generate a small synthetic CSV mimicking CICIDS2017 column names for pipeline smoke tests.
NOT for thesis results — only verifies code paths before real data is downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (
    COHERENCE_ANCHOR_FEATURES,
    CONTINUOUS_FEATURES,
    DISCRETE_INTEGER_FEATURES,
    IMMUTABLE_FEATURES,
    LEAKAGE_COLUMNS,
)

LABEL_COL = "Label"
N_ROWS = 60_000
SEED = 42


def main() -> None:
    rng = np.random.default_rng(SEED)
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(
        dict.fromkeys(
            IMMUTABLE_FEATURES
            + COHERENCE_ANCHOR_FEATURES
            + DISCRETE_INTEGER_FEATURES
            + CONTINUOUS_FEATURES
        )
    )
    classes = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
        "Infiltration",
        "Web Attack",
    ]
    y = rng.choice(classes, size=N_ROWS, p=[0.5, 0.15, 0.15, 0.1, 0.05, 0.05])

    data: dict = {LABEL_COL: y}
    for c in LEAKAGE_COLUMNS:
        if c == "Timestamp":
            data[c] = np.arange(N_ROWS)
        else:
            data[c] = rng.integers(0, 1000, size=N_ROWS)

    for c in IMMUTABLE_FEATURES:
        if c == "Destination Port":
            data[c] = rng.integers(1, 65535, size=N_ROWS)
        elif c == "Protocol":
            data[c] = rng.choice([6, 17], size=N_ROWS)
        else:
            data[c] = rng.integers(0, 100, size=N_ROWS)

    data["Flow Duration"] = rng.exponential(1000, size=N_ROWS)
    for c in DISCRETE_INTEGER_FEATURES:
        data[c] = rng.integers(0, 500, size=N_ROWS)
    for c in CONTINUOUS_FEATURES:
        data[c] = rng.normal(0, 1, size=N_ROWS).astype(np.float64)

    df = pd.DataFrame(data)
    out_path = out_dir / "synthetic_smoke.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
