#!/usr/bin/env python3
"""Run headline metrics (constrained PGD eps=0.01) across multiple test subsample seeds."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config
from src.constraints_torch import make_constraint_fn
from src.io_utils import load_mlp_checkpoint, load_processed_data, load_scaler_label_encoder
from src.models.mlp import predict_mlp
from src.attack_metrics import evaluate_attack
from scripts.run_attacks import resolve_test_cap, subsample_test
from src.attacks.pgd import pgd_attack

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch required") from exc


def _headline_pgd_constrained(
    cfg: dict,
    *,
    seed: int,
    baseline_run: str | None,
) -> dict:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    splits, metadata = load_processed_data(processed_dir)
    label_names = metadata["class_names"]
    max_test = resolve_test_cap(cfg, pilot=False)
    X_test, y_test = subsample_test(
        splits["X_test"], splits["y_test"], max_test, seed
    )
    scaler, _ = load_scaler_label_encoder(processed_dir)
    feature_groups = metadata.get("feature_groups", {})
    cfn = make_constraint_fn(
        metadata["feature_names"], feature_groups, cfg, scaler=scaler
    )
    acfg = cfg["attacks"]
    epsilon = 0.01
    pgd_cfg = acfg["pgd"]
    steps = int(pgd_cfg.get("steps_eval", pgd_cfg["steps"]))
    alpha = float(pgd_cfg["alpha"])
    batch_size = int(acfg.get("batch_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if baseline_run:
        ckpt_path = results_dir / "baselines" / baseline_run / "mlp_model.pt"
    else:
        from src.io_utils import find_latest_baseline_run

        ckpt_path = find_latest_baseline_run(results_dir) / "mlp_model.pt"
    model, _ = load_mlp_checkpoint(ckpt_path, device)

    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).long()
    y_clean_pred, _ = predict_mlp(model, X_test, device)
    x_adv = pgd_attack(
        model,
        x_tensor,
        y_tensor,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        device=device,
        random_start=True,
        constraint_fn=cfn,
        batch_size=batch_size,
    )
    y_adv_pred, _ = predict_mlp(model, x_adv.cpu().numpy(), device)
    metrics = evaluate_attack(
        (y_clean_pred, y_adv_pred), y_test, X_test, x_adv.cpu().numpy(), label_names
    )
    return {
        "seed": seed,
        "n_test": int(len(y_test)),
        "baseline_run": ckpt_path.parent.name,
        "attack": "pgd",
        "mode": "constrained",
        "epsilon": epsilon,
        "pgd_steps": steps,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed headline constrained PGD.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--baseline-run", default="20260520T192855Z")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    rows = []
    for seed in args.seeds:
        cfg_run = deepcopy(cfg)
        cfg_run["seed"] = seed
        print(f"\n=== seed {seed} ===")
        row = _headline_pgd_constrained(
            cfg_run, seed=seed, baseline_run=args.baseline_run
        )
        m = row["metrics"]
        print(f"  ASR={m['asr']:.4f}  robust_acc={m['robust_accuracy']:.4f}")
        rows.append(row)

    asrs = [r["metrics"]["asr"] for r in rows]
    robs = [r["metrics"]["robust_accuracy"] for r in rows]
    summary = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "headline": "pgd_constrained_eps0.01",
        "seeds": args.seeds,
        "mean_asr": float(np.mean(asrs)),
        "std_asr": float(np.std(asrs)),
        "mean_robust_accuracy": float(np.mean(robs)),
        "std_robust_accuracy": float(np.std(robs)),
        "per_seed": rows,
    }
    out_dir = Path(cfg["paths"]["results_dir"]) / "multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"headline_{summary['run_id']}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nMean ASR {summary['mean_asr']:.4f} ± {summary['std_asr']:.4f}")
    print(f"Mean robust acc {summary['mean_robust_accuracy']:.4f} ± {summary['std_robust_accuracy']:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
