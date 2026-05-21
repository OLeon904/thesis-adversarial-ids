#!/usr/bin/env python3
"""Evaluate FGSM/PGD on adversarially trained MLP checkpoints (each pass)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attack_metrics import evaluate_attack, save_attack_results
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.config import ensure_dirs, load_config
from src.constraints_torch import make_constraint_fn
from src.io_utils import load_mlp_checkpoint, load_processed_data, load_scaler_label_encoder
from src.models.mlp import predict_mlp
from scripts.run_attacks import resolve_test_cap, subsample_test, modes_to_run, _epsilon_tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-run", required=True, help="Run id under results/adv_train/")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--mode", default="both", choices=["unconstrained", "constrained", "both"])
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    splits, metadata = load_processed_data(processed_dir)
    label_names = metadata["class_names"]

    max_test = resolve_test_cap(cfg, args.pilot)
    X_test, y_test = subsample_test(splits["X_test"], splits["y_test"], max_test, cfg["seed"])

    feature_groups = metadata.get("feature_groups", {})
    scaler, _ = load_scaler_label_encoder(processed_dir)
    constraint_fn = make_constraint_fn(
        metadata["feature_names"], feature_groups, cfg, scaler=scaler
    )
    acfg = cfg["attacks"]
    pgd_cfg = acfg["pgd"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adv_dir = Path(cfg["paths"]["results_dir"]) / "adv_train" / args.adv_run
    out_dir = Path(cfg["paths"]["results_dir"]) / "attacks" / f"adv_eval_{args.adv_run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).long()
    rows = []

    for p in range(1, args.passes + 1):
        ckpt = adv_dir / f"pass_{p}.pt"
        if not ckpt.is_file():
            print(f"Skip missing {ckpt}")
            continue
        model, _ = load_mlp_checkpoint(ckpt, device)
        y_clean_pred, _ = predict_mlp(model, X_test, device)

        for attack_name in ("fgsm", "pgd"):
            for attack_mode in modes_to_run(args.mode):
                cfn = constraint_fn if attack_mode == "constrained" else None
                for epsilon in acfg["epsilon_values"]:
                    if attack_name == "fgsm":
                        x_adv_t = fgsm_attack(
                            model, x_tensor, y_tensor, epsilon, device,
                            constraint_fn=cfn, chunk_size=acfg["batch_size"],
                        )
                    else:
                        x_adv_t = pgd_attack(
                            model, x_tensor, y_tensor, epsilon,
                            pgd_cfg["alpha"], pgd_cfg.get("steps_eval", pgd_cfg["steps"]),
                            device, constraint_fn=cfn, batch_size=acfg["batch_size"],
                        )
                    X_adv = x_adv_t.cpu().numpy()
                    y_adv_pred, _ = predict_mlp(model, X_adv, device)
                    metrics = evaluate_attack(
                        (y_clean_pred, y_adv_pred), y_test, X_test, X_adv, label_names
                    )
                    rec = {
                        "adv_pass": p,
                        "attack": attack_name,
                        "mode": attack_mode,
                        "epsilon": epsilon,
                        "metrics": metrics,
                    }
                    path = out_dir / f"pass{p}_{attack_name}_{attack_mode}_{_epsilon_tag(epsilon)}.json"
                    save_attack_results(rec, path)
                    rows.append(rec)
                    print(
                        f"pass={p} {attack_name} {attack_mode} eps={epsilon} "
                        f"ASR={metrics['asr']:.4f} robust={metrics['robust_accuracy']:.4f}"
                    )

    summary = {
        "adv_run": args.adv_run,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "results": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
