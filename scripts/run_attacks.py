#!/usr/bin/env python3
"""Run FGSM/PGD adversarial attacks on the baseline MLP (unconstrained and constrained)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ensure_dirs, load_config, save_config_snapshot
from src.io_utils import load_scaler_label_encoder
from src.models.mlp import predict_mlp

USAGE_HELP = """
Run FGSM and PGD on the baseline MLP test set.

Examples:
  py -3 scripts/run_attacks.py --pilot
  py -3 scripts/run_attacks.py --mode both
  py -3 scripts/run_attacks.py --checkpoint results/baselines/<run_id>/mlp_model.pt
  py -3 scripts/run_attacks.py --mode constrained --pilot

Options:
  --pilot       Cap test rows via attacks.max_test_samples in config
  --mode        unconstrained | constrained | both (default: both)
  --checkpoint  Path to mlp_model.pt (default: latest baseline run)
  --config      Alternate YAML config path

Outputs JSON metrics under results/attacks/<run_id>/ per attack, mode, and epsilon.
""".strip()

# --- optional attack modules (inline fallbacks if missing) ---

try:
    from src.attacks.fgsm import fgsm_attack
except ImportError:
    import torch.nn as nn
    import torch.nn.functional as F

    def fgsm_attack(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
        device: torch.device,
        targeted: bool = False,
        constraint_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        x = x.to(device)
        y = y.to(device).long()
        x_orig = x.detach()
        x_adv = x_orig.clone().requires_grad_(True)
        model.eval()
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        sign = x_adv.grad.detach().sign()
        delta = -epsilon * sign if targeted else epsilon * sign
        x_adv = torch.clamp(x_orig + delta, x_orig - epsilon, x_orig + epsilon)
        if constraint_fn is not None:
            x_adv = constraint_fn(x_adv, x_orig)
        return x_adv.detach()


try:
    from src.attacks.pgd import pgd_attack
except ImportError:
    import torch.nn as nn
    import torch.nn.functional as F

    def _project_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, epsilon: float) -> torch.Tensor:
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        return x_orig + delta

    def pgd_attack(
        model: nn.Module,
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        epsilon: float,
        alpha: float,
        steps: int,
        device: torch.device,
        random_start: bool = True,
        constraint_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        batch_size: int | None = 256,
    ) -> torch.Tensor:
        x_t = torch.from_numpy(x).float().to(device) if isinstance(x, np.ndarray) else x.to(device)
        y_t = (
            torch.from_numpy(y).long().to(device)
            if isinstance(y, np.ndarray)
            else y.to(device).long()
        )
        bs = batch_size or x_t.shape[0]
        chunks: list[torch.Tensor] = []
        for start in range(0, x_t.shape[0], bs):
            end = min(start + bs, x_t.shape[0])
            xc, yc = x_t[start:end], y_t[start:end]
            x_orig = xc.detach()
            x_adv = x_orig.clone()
            if random_start:
                x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
                x_adv = _project_linf(x_adv, x_orig, epsilon)
                if constraint_fn is not None:
                    x_adv = constraint_fn(x_adv, x_orig)
            model.eval()
            for _ in range(steps):
                x_adv = x_adv.detach().requires_grad_(True)
                loss = F.cross_entropy(model(x_adv), yc)
                grad = torch.autograd.grad(loss, x_adv)[0]
                with torch.no_grad():
                    x_adv = _project_linf(x_adv + alpha * grad.sign(), x_orig, epsilon)
                    if constraint_fn is not None:
                        x_adv = constraint_fn(x_adv, x_orig)
            chunks.append(x_adv.detach())
        return torch.cat(chunks, dim=0)


try:
    from src.constraints_torch import make_constraint_fn
except ImportError:

    def make_constraint_fn(
        feature_names: list[str],
        feature_groups: dict[str, list[str]],
        max_perturbation_ratio: float,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        def _indices(group: str) -> list[int]:
            return [name_to_idx[n] for n in feature_groups.get(group, []) if n in name_to_idx]

        imm = _indices("immutable")
        anchor = _indices("coherence_anchor")
        discrete = _indices("discrete_integer")

        def constraint_fn(x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
            out = x_adv.clone()
            if imm:
                out[:, imm] = x_orig[:, imm]
            if anchor:
                out[:, anchor] = x_orig[:, anchor]
            if discrete:
                out[:, discrete] = torch.round(out[:, discrete])
            if max_perturbation_ratio > 0:
                delta = out - x_orig
                cap = max_perturbation_ratio * (x_orig.abs() + 1e-8)
                delta = torch.clamp(delta, -cap, cap)
                out = x_orig + delta
                if imm:
                    out[:, imm] = x_orig[:, imm]
                if anchor:
                    out[:, anchor] = x_orig[:, anchor]
            return out

        return constraint_fn


try:
    from src.attacks.attack_metrics import evaluate_attack, save_attack_results
except ImportError:
    try:
        from src.attack_metrics import evaluate_attack, save_attack_results
    except ImportError:
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        def evaluate_attack(
            model_or_preds: Any,
            y_true: np.ndarray,
            y_clean: np.ndarray,
            y_adv: np.ndarray,
            label_names: list[str],
        ) -> dict[str, Any]:
            if isinstance(model_or_preds, (tuple, list)):
                y_clean_pred, y_adv_pred = model_or_preds
            else:
                y_clean_pred = model_or_preds.predict(y_clean)
                y_adv_pred = model_or_preds.predict(y_adv)
            y_true = np.asarray(y_true)
            clean_correct = y_clean_pred == y_true
            n_cc = int(clean_correct.sum())
            asr = (
                float((clean_correct & (y_adv_pred != y_true)).sum() / n_cc)
                if n_cc
                else 0.0
            )
            report = classification_report(
                y_true, y_adv_pred, target_names=label_names, output_dict=True, zero_division=0
            )
            per_class = {
                name: float(report[name]["recall"])
                for name in label_names
                if name in report
            }
            return {
                "accuracy": float(accuracy_score(y_true, y_clean_pred)),
                "macro_f1": float(
                    f1_score(y_true, y_adv_pred, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_true, y_adv_pred, average="weighted", zero_division=0)
                ),
                "asr": asr,
                "robust_accuracy": float(accuracy_score(y_true, y_adv_pred)),
                "per_class_recall": per_class,
            }

        def save_attack_results(metrics: dict[str, Any], path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)


try:
    from src.attacks.io_utils import (
        find_latest_baseline_run,
        load_mlp_checkpoint,
        load_processed_data,
    )
except ImportError:
    from src.io_utils import (
        find_latest_baseline_run,
        load_mlp_checkpoint,
        load_processed_data,
    )


def _epsilon_tag(epsilon: float) -> str:
    return f"eps{epsilon:g}".replace(".", "p")


def resolve_checkpoint(cfg: dict[str, Any], checkpoint_arg: str | None) -> tuple[Path, str | None]:
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            path = ROOT / path
        return path.resolve(), None

    results_dir = Path(cfg["paths"]["results_dir"])
    baseline_id = cfg["paths"].get("baseline_run")
    if baseline_id:
        run_dir = results_dir / "baselines" / baseline_id
    else:
        run_dir = find_latest_baseline_run(results_dir)

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        mlp_path = Path(summary["models"]["mlp"]["model_path"])
        return mlp_path.resolve(), summary.get("run_id", run_dir.name)

    mlp_path = run_dir / "mlp_model.pt"
    if not mlp_path.is_file():
        raise FileNotFoundError(f"No MLP checkpoint in {run_dir}")
    return mlp_path.resolve(), run_dir.name


def subsample_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y_test) <= max_samples:
        return X_test, y_test
    X_sub, _, y_sub, _ = train_test_split(
        X_test,
        y_test,
        train_size=max_samples,
        stratify=y_test,
        random_state=seed,
    )
    print(f"  Test subsample: {len(y_sub):,} / {len(y_test):,} rows")
    return X_sub, y_sub


def resolve_test_cap(cfg: dict[str, Any], pilot: bool) -> int | None:
    cap = cfg["attacks"].get("max_test_samples")
    if pilot:
        return cap if cap is not None else 10_000
    return cap


def modes_to_run(mode: str) -> list[str]:
    if mode == "both":
        return ["unconstrained", "constrained"]
    if mode not in ("unconstrained", "constrained"):
        raise ValueError(f"Invalid --mode {mode!r}; use unconstrained, constrained, or both")
    return [mode]


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = ("Attack", "Mode", "Epsilon", "ASR", "Robust Acc", "Macro F1")
    print("\n" + "═" * 72)
    print("Attack summary")
    print("═" * 72)
    print(
        f"{headers[0]:<8} {headers[1]:<14} {headers[2]:>10} "
        f"{headers[3]:>8} {headers[4]:>12} {headers[5]:>10}"
    )
    print("-" * 72)
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['attack']:<8} {r['mode']:<14} {r['epsilon']:>10.4g} "
            f"{m['asr']:>8.4f} {m['robust_accuracy']:>12.4f} {m['macro_f1']:>10.4f}"
        )
    print("═" * 72)


def run_attacks(
    cfg: dict[str, Any],
    *,
    pilot: bool = False,
    mode: str = "both",
    checkpoint: str | None = None,
    config_path: Path | str | None = None,
) -> Path:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    splits, metadata = load_processed_data(processed_dir)
    label_names = metadata["class_names"]

    ckpt_path, baseline_run_id = resolve_checkpoint(cfg, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ckpt_meta = load_mlp_checkpoint(ckpt_path, device)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    if baseline_run_id:
        print(f"Baseline run: {baseline_run_id}")

    max_test = resolve_test_cap(cfg, pilot)
    X_test, y_test = subsample_test(
        splits["X_test"], splits["y_test"], max_test, cfg["seed"]
    )

    acfg = cfg["attacks"]
    pgd_cfg = acfg["pgd"]
    epsilon_values = acfg["epsilon_values"]
    batch_size = acfg.get("batch_size", 256)
    pgd_alpha = pgd_cfg["alpha"]
    pgd_steps = pgd_cfg.get("steps_eval", pgd_cfg["steps"])

    feature_groups = metadata.get("feature_groups")
    if not feature_groups:
        from src.constraints import load_feature_groups

        feature_groups = load_feature_groups(processed_dir / "metadata.json")
    scaler, _ = load_scaler_label_encoder(processed_dir)
    constraint_fn = make_constraint_fn(
        metadata["feature_names"],
        feature_groups,
        cfg,
        scaler=scaler,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if pilot:
        run_id = f"pilot_{run_id}"
    out_dir = results_dir / "attacks" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = save_config_snapshot(out_dir, config_path)

    x_tensor = torch.from_numpy(X_test).float()
    y_tensor = torch.from_numpy(y_test).long()
    y_clean_pred, _ = predict_mlp(model, X_test, device)

    summary_rows: list[dict[str, Any]] = []
    attack_modes = modes_to_run(mode)

    for attack_name in ("fgsm", "pgd"):
        for attack_mode in attack_modes:
            cfn = constraint_fn if attack_mode == "constrained" else None
            for epsilon in epsilon_values:
                print(f"\n{attack_name.upper()}  mode={attack_mode}  epsilon={epsilon}")

                if attack_name == "fgsm":
                    x_adv_t = fgsm_attack(
                        model,
                        x_tensor,
                        y_tensor,
                        epsilon,
                        device,
                        constraint_fn=cfn,
                        chunk_size=batch_size,
                    )
                else:
                    x_adv_t = pgd_attack(
                        model,
                        x_tensor,
                        y_tensor,
                        epsilon,
                        pgd_alpha,
                        pgd_steps,
                        device,
                        random_start=True,
                        constraint_fn=cfn,
                        batch_size=batch_size,
                    )

                X_adv = x_adv_t.cpu().numpy()
                y_adv_pred, _ = predict_mlp(model, X_adv, device)
                metrics = evaluate_attack(
                    (y_clean_pred, y_adv_pred),
                    y_test,
                    X_test,
                    X_adv,
                    label_names,
                )

                record: dict[str, Any] = {
                    "run_id": run_id,
                    "pilot": pilot,
                    "attack": attack_name,
                    "mode": attack_mode,
                    "epsilon": epsilon,
                    "checkpoint": str(ckpt_path),
                    "baseline_run_id": baseline_run_id,
                    "n_test_samples": int(len(y_test)),
                    "device": str(device),
                    "metrics": metrics,
                }
                if attack_name == "pgd":
                    record["pgd_alpha"] = pgd_alpha
                    record["pgd_steps"] = pgd_steps

                out_path = out_dir / f"{attack_name}_{attack_mode}_{_epsilon_tag(epsilon)}.json"
                save_attack_results(record, out_path)
                print(f"  saved {out_path.name}")
                print(
                    f"  ASR={metrics['asr']:.4f}  "
                    f"robust_acc={metrics['robust_accuracy']:.4f}  "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )
                summary_rows.append(
                    {
                        "attack": attack_name,
                        "mode": attack_mode,
                        "epsilon": epsilon,
                        "metrics": metrics,
                        "path": str(out_path),
                    }
                )

    manifest = {
        "run_id": run_id,
        "config_snapshot": str(config_snapshot),
        "pilot": pilot,
        "mode_arg": mode,
        "checkpoint": str(ckpt_path),
        "baseline_run_id": baseline_run_id,
        "n_test_samples": int(len(y_test)),
        "n_test_total": int(len(splits["y_test"])),
        "test_subsample": max_test is not None and len(y_test) < len(splits["y_test"]),
        "model_input_space": metadata.get("model_input_space", "standardized"),
        "constraint_projection_space": metadata.get(
            "constraint_projection_space", "raw_physical"
        ),
        "epsilon_values": epsilon_values,
        "results": [
            {
                "attack": r["attack"],
                "mode": r["mode"],
                "epsilon": r["epsilon"],
                "path": r["path"],
                "asr": r["metrics"]["asr"],
                "robust_accuracy": r["metrics"]["robust_accuracy"],
            }
            for r in summary_rows
        ],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print_summary_table(summary_rows)
    print(f"\nResults saved to {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FGSM/PGD attacks on the baseline MLP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_HELP,
    )
    parser.add_argument("--pilot", action="store_true", help="Cap test set (attacks.max_test_samples)")
    parser.add_argument(
        "--mode",
        choices=("unconstrained", "constrained", "both"),
        default="both",
        help="Attack constraint mode (default: both)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to mlp_model.pt (default: latest baseline run)",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    ensure_dirs(cfg)
    run_attacks(
        cfg,
        pilot=args.pilot,
        mode=args.mode,
        checkpoint=args.checkpoint,
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
