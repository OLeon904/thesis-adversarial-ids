#!/usr/bin/env python3
"""Audit constraint violations before/after raw-space projection on scaled L∞ perturbations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constraint_violations import count_raw_violations, summarize_violations
from src.constraints import project_batch
from src.features import classify_columns
from src.io_utils import load_processed_data, load_scaler_label_encoder


def _load_test_batch(
    cfg: dict, n_samples: int, seed: int
) -> tuple[np.ndarray, list[str], object]:
    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"])
    splits, meta = load_processed_data(processed_dir)
    x_test = splits["X_test"]
    rng = np.random.default_rng(seed)
    if n_samples < len(x_test):
        idx = rng.choice(len(x_test), size=n_samples, replace=False)
        x_test = x_test[idx]
    scaler, _ = load_scaler_label_encoder(processed_dir)
    return x_test.astype(np.float32), meta["feature_names"], scaler


def audit_batch(
    x_scaled: np.ndarray,
    *,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_names: list[str],
    groups: dict[str, list[str]],
    cfg: dict,
    epsilon: float = 0.01,
    seed: int = 42,
) -> dict:
    """Simulate one scaled L∞ step at ε, decode to raw, count violations, project, recount."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale**2
    scaler.n_features_in_ = len(scaler_mean)

    rng = np.random.default_rng(seed)
    delta = rng.uniform(-epsilon, epsilon, size=x_scaled.shape).astype(np.float32)
    x_perturbed_scaled = np.clip(
        x_scaled.astype(np.float64) + delta,
        x_scaled - epsilon,
        x_scaled + epsilon,
    )
    xa_raw = scaler.inverse_transform(x_perturbed_scaled)
    xo_raw = scaler.inverse_transform(x_scaled)
    ccfg = cfg.get("constraints", {})
    mode = ccfg.get("mode", "relaxed")
    max_ratio = float(ccfg.get("max_perturbation_ratio", 0.20))

    before = count_raw_violations(
        xa_raw, xo_raw, feature_names=feature_names, groups=groups, max_perturbation_ratio=max_ratio
    )
    projected_raw = project_batch(
        xa_raw,
        xo_raw,
        groups,
        feature_names=feature_names,
        mode=mode,
        max_perturbation_ratio=max_ratio,
    )
    after = count_raw_violations(
        projected_raw,
        xo_raw,
        feature_names=feature_names,
        groups=groups,
        max_perturbation_ratio=max_ratio,
    )
    return {
        "n_samples": int(x_scaled.shape[0]),
        "n_features": int(x_scaled.shape[1]),
        "epsilon_simulated": epsilon,
        "constraints_mode": mode,
        "before_projection": summarize_violations(before),
        "after_projection": summarize_violations(after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw constraint violations.")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    x_test, feature_names, scaler = _load_test_batch(cfg, args.n_samples, args.seed)
    groups = classify_columns(feature_names)

    report = audit_batch(
        x_test,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        feature_names=feature_names,
        groups=groups,
        cfg=cfg,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    report["feature_groups"] = {k: len(v) for k, v in groups.items()}

    out = args.out or Path(cfg["paths"]["results_dir"]) / "constraint_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
