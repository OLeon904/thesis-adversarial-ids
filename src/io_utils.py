"""Load processed datasets, sklearn/torch checkpoints, and baseline run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.models.mlp import MLPClassifier

__all__ = [
    "load_processed_data",
    "load_mlp_checkpoint",
    "load_rf_checkpoint",
    "find_latest_baseline_run",
    "load_scaler_label_encoder",
]


def _as_path(path: Path | str) -> Path:
    return Path(path)


def load_processed_data(
    processed_dir: Path | str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load train/val/test splits and preprocessing metadata.

    Expects ``splits.npz`` and ``metadata.json`` under *processed_dir*.
    """
    root = _as_path(processed_dir)
    splits_npz = np.load(root / "splits.npz")
    splits = {
        "X_train": splits_npz["X_train"],
        "X_val": splits_npz["X_val"],
        "X_test": splits_npz["X_test"],
        "y_train": splits_npz["y_train"],
        "y_val": splits_npz["y_val"],
        "y_test": splits_npz["y_test"],
    }
    with (root / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return splits, metadata


def load_scaler_label_encoder(
    processed_dir: Path | str,
) -> tuple[StandardScaler, LabelEncoder]:
    """Load fitted ``scaler.joblib`` and ``label_encoder.joblib``."""
    root = _as_path(processed_dir)
    scaler = joblib.load(root / "scaler.joblib")
    label_encoder = joblib.load(root / "label_encoder.joblib")
    return scaler, label_encoder


def load_mlp_checkpoint(
    path: Path | str,
    device: torch.device | str,
) -> tuple[MLPClassifier, dict[str, Any]]:
    """Restore an MLP from a ``mlp_model.pt`` checkpoint written by ``train_mlp``."""
    ckpt_path = _as_path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"MLP checkpoint not found: {ckpt_path}")

    dev = torch.device(device) if isinstance(device, str) else device
    raw = torch.load(ckpt_path, map_location=dev, weights_only=False)

    model = MLPClassifier(
        n_features=raw["n_features"],
        n_classes=raw["n_classes"],
        hidden_dims=raw["hidden_dims"],
        dropout=raw["dropout"],
    ).to(dev)
    model.load_state_dict(raw["state_dict"])
    model.eval()

    metadata = {
        k: v for k, v in raw.items() if k != "state_dict"
    }
    return model, metadata


def load_rf_checkpoint(path: Path | str) -> Any:
    """Load a random forest saved with joblib (e.g. ``rf_model.joblib``)."""
    ckpt_path = _as_path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"RF checkpoint not found: {ckpt_path}")
    return joblib.load(ckpt_path)


def find_latest_baseline_run(results_dir: Path | str) -> Path:
    """Return the most recently modified directory under ``results/baselines/``."""
    baselines_root = _as_path(results_dir) / "baselines"
    if not baselines_root.is_dir():
        raise FileNotFoundError(f"Baselines directory not found: {baselines_root}")

    runs = [p for p in baselines_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No baseline runs in {baselines_root}")

    return max(runs, key=lambda p: p.stat().st_mtime)
