#!/usr/bin/env python3
"""
Fast end-to-end smoke test for preprocess → baseline → attack code paths.

Uses pilot sizes (512 test samples, tiny attack batches). Target: <2 min on CPU.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.config import load_config
from src.io_utils import (
    find_latest_baseline_run,
    load_mlp_checkpoint,
    load_processed_data,
)

PILOT_TEST_SAMPLES = 512
FGSM_BATCH = 16
PGD_BATCH = 16
PGD_STEPS = 2


def _report(step: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {step}{suffix}")
    return ok


def _find_mlp_checkpoint(results_dir: Path) -> Path | None:
    baselines_root = results_dir / "baselines"
    if not baselines_root.is_dir():
        return None
    runs = sorted(
        (p for p in baselines_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in runs:
        ckpt = run_dir / "mlp_model.pt"
        if ckpt.is_file():
            return ckpt
    return None


def _load_constraint_fn(
    metadata: dict,
    cfg: dict,
    processed_dir: Path,
) -> tuple[object | None, str]:
    """Return raw-space constraint_fn (requires scaler + constraints_torch)."""
    try:
        from src.constraints_torch import make_constraint_fn
        from src.io_utils import load_scaler_label_encoder
    except Exception as exc:
        return None, f"import failed: {exc}"

    feature_names = metadata.get("feature_names", [])
    feature_groups = metadata.get("feature_groups")
    if not feature_groups:
        from src.constraints import load_feature_groups

        feature_groups = load_feature_groups(processed_dir / "metadata.json")

    scaler, _ = load_scaler_label_encoder(processed_dir)
    fn = make_constraint_fn(
        feature_names,
        feature_groups,
        cfg,
        scaler=scaler,
    )
    return fn, "raw_physical via make_constraint_fn(scaler=...)"


def main() -> int:
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    splits_path = processed_dir / "splits.npz"

    all_ok = True

    # 1. Verify splits.npz
    try:
        ok = splits_path.is_file()
        all_ok &= _report(
            "splits.npz exists",
            ok,
            str(splits_path) if ok else f"missing: {splits_path}",
        )
        if not ok:
            return 1
    except Exception as exc:
        all_ok &= _report("splits.npz exists", False, str(exc))
        return 1

    # 2. Load pilot test samples
    try:
        splits, metadata = load_processed_data(processed_dir)
        n = min(PILOT_TEST_SAMPLES, len(splits["y_test"]))
        X_test = splits["X_test"][:n].astype(np.float32)
        y_test = splits["y_test"][:n]
        all_ok &= _report(
            "load test samples",
            True,
            f"{n} rows, {X_test.shape[1]} features",
        )
    except Exception as exc:
        all_ok &= _report("load test samples", False, str(exc))
        return 1

    device = torch.device("cpu")
    attacks_cfg = cfg["attacks"]
    epsilon = float(attacks_cfg["epsilon_values"][0])
    alpha = float(attacks_cfg["pgd"]["alpha"])

    # 3. Load latest baseline MLP (optional)
    model = None
    mlp_path = _find_mlp_checkpoint(results_dir)
    if mlp_path is None:
        try:
            run_dir = find_latest_baseline_run(results_dir)
            candidate = run_dir / "mlp_model.pt"
            if candidate.is_file():
                mlp_path = candidate
        except FileNotFoundError:
            pass

    if mlp_path is None:
        _report(
            "load baseline MLP",
            True,
            "SKIP — no mlp_model.pt under results/baselines/",
        )
        _report("FGSM (1 step, unconstrained)", True, "SKIP — no model")
        _report("PGD (2 steps)", True, "SKIP — no model")
        print("\nSmoke test finished (data path OK; train baselines for attack checks).")
        return 0 if all_ok else 1

    try:
        model, _ = load_mlp_checkpoint(mlp_path, device)
        all_ok &= _report("load baseline MLP", True, str(mlp_path))
    except Exception as exc:
        all_ok &= _report("load baseline MLP", False, str(exc))
        _report("FGSM (1 step, unconstrained)", True, "SKIP — model load failed")
        _report("PGD (2 steps)", True, "SKIP — model load failed")
        return 1

    x_tiny = torch.from_numpy(X_test[:FGSM_BATCH])
    y_tiny = torch.from_numpy(y_test[:FGSM_BATCH]).long()

    # 4. FGSM — one step, unconstrained, tiny batch
    try:
        x_adv = fgsm_attack(
            model,
            x_tiny,
            y_tiny,
            epsilon=epsilon,
            device=device,
            constraint_fn=None,
            chunk_size=FGSM_BATCH,
        )
        ok = (
            x_adv.shape == x_tiny.shape
            and torch.isfinite(x_adv).all()
            and not torch.allclose(x_adv, x_tiny)
        )
        delta = (x_adv - x_tiny).abs().max().item()
        all_ok &= _report(
            "FGSM (1 step, unconstrained)",
            ok,
            f"batch={FGSM_BATCH}, eps={epsilon}, max|delta|={delta:.6f}",
        )
    except Exception as exc:
        all_ok &= _report("FGSM (1 step, unconstrained)", False, str(exc))
        traceback.print_exc()

    # 5. PGD — 2 steps; constrained when constraints.py exists
    try:
        constraint_fn, constraint_detail = _load_constraint_fn(
            metadata, cfg, processed_dir
        )
        x_pgd = torch.from_numpy(X_test[:PGD_BATCH])
        y_pgd = torch.from_numpy(y_test[:PGD_BATCH]).long()

        x_adv_pgd = pgd_attack(
            model,
            x_pgd,
            y_pgd,
            epsilon=epsilon,
            alpha=alpha,
            steps=PGD_STEPS,
            device=device,
            random_start=False,
            constraint_fn=constraint_fn,
            batch_size=PGD_BATCH,
        )
        ok = (
            x_adv_pgd.shape == x_pgd.shape
            and torch.isfinite(x_adv_pgd).all()
        )
        mode = "constrained" if constraint_fn is not None else "unconstrained"
        all_ok &= _report(
            f"PGD ({PGD_STEPS} steps, {mode})",
            ok,
            f"batch={PGD_BATCH}, eps={epsilon}, alpha={alpha}; {constraint_detail}",
        )
    except Exception as exc:
        all_ok &= _report(f"PGD ({PGD_STEPS} steps)", False, str(exc))
        traceback.print_exc()

    print()
    if all_ok:
        print("Smoke test: all steps passed.")
    else:
        print("Smoke test: one or more steps failed.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
