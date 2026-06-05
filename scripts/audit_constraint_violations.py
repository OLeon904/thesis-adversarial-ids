#!/usr/bin/env python3
"""Audit constraint violations before/after raw-space projection on attack outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constraint_violations import audit_scaled_adv_batch
from src.constraints_torch import make_constraint_fn
from src.features import classify_columns
from src.io_utils import load_processed_data, load_scaler_label_encoder, load_mlp_checkpoint
from src.attacks.pgd import pgd_attack

from scripts.run_attacks import (
    _epsilon_tag,
    resolve_checkpoint,
    subsample_test,
)


def _load_test_batch(
    cfg: dict, n_samples: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str], object, dict[str, list[str]]]:
    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"])
    splits, meta = load_processed_data(processed_dir)
    max_test = cfg["attacks"].get("max_test_samples")
    x_test, y_test = subsample_test(splits["X_test"], splits["y_test"], max_test, seed)
    if n_samples is not None and n_samples < len(y_test):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_test), size=n_samples, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]
    scaler, _ = load_scaler_label_encoder(processed_dir)
    groups = meta.get("feature_groups") or classify_columns(meta["feature_names"])
    return (
        x_test.astype(np.float32),
        y_test,
        meta["feature_names"],
        scaler,
        groups,
    )


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
    return audit_scaled_adv_batch(
        x_perturbed_scaled.astype(np.float32),
        x_scaled,
        scaler=scaler,
        feature_names=feature_names,
        groups=groups,
        cfg=cfg,
    ) | {
        "epsilon_simulated": epsilon,
        "audit_kind": "random_linf_step",
    }


def discover_x_adv_file(
    run_dir: Path,
    attack: str,
    attack_mode: str,
    epsilon: float,
) -> Path | None:
    tag = _epsilon_tag(epsilon)
    candidates = [
        run_dir / f"x_adv_{attack}_{attack_mode}_{tag}.npy",
        run_dir / f"x_adv_{attack}_{attack_mode}_eps{epsilon:g}.npy",
        run_dir / f"{attack}_{attack_mode}_{tag}_x_adv.npy",
        run_dir / f"{attack}_{attack_mode}_{tag}.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(run_dir.glob(f"*x_adv*{tag}*")):
        if path.suffix in (".npy", ".npz"):
            return path
    return None


def load_x_adv_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "x_adv" not in data:
            raise KeyError(f"Expected key 'x_adv' in {path}, got {list(data.files)}")
        return np.asarray(data["x_adv"], dtype=np.float32)
    return np.load(path).astype(np.float32)


def generate_constrained_adv(
    cfg: dict,
    *,
    checkpoint: str | Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack: str,
    attack_mode: str,
    epsilon: float,
    n_samples: int,
    one_batch: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path, baseline_run_id = resolve_checkpoint(cfg, str(checkpoint))
    model, _ = load_mlp_checkpoint(ckpt_path, device)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    _, meta = load_processed_data(processed_dir)
    scaler, _ = load_scaler_label_encoder(processed_dir)
    groups = meta.get("feature_groups") or classify_columns(meta["feature_names"])
    constraint_fn = make_constraint_fn(
        meta["feature_names"],
        groups,
        cfg,
        scaler=scaler,
    )

    acfg = cfg["attacks"]
    batch_size = acfg.get("batch_size", 256)
    if one_batch:
        n = min(batch_size, len(y_test), n_samples)
    else:
        n = min(n_samples, len(y_test))

    x_sub = x_test[:n]
    y_sub = y_test[:n]
    x_tensor = torch.from_numpy(x_sub).float()
    y_tensor = torch.from_numpy(y_sub).long()
    cfn = constraint_fn if attack_mode == "constrained" else None

    meta_out = {
        "checkpoint": str(ckpt_path),
        "baseline_run_id": baseline_run_id,
        "device": str(device),
        "n_generated": n,
        "one_batch": one_batch,
        "attack_mode": attack_mode,
    }

    if attack == "pgd":
        pgd_cfg = acfg["pgd"]
        alpha = pgd_cfg["alpha"]
        steps = pgd_cfg.get("steps_eval", pgd_cfg["steps"])
        meta_out["pgd_alpha"] = alpha
        meta_out["pgd_steps"] = steps
        x_adv_t = pgd_attack(
            model,
            x_tensor,
            y_tensor,
            epsilon,
            alpha,
            steps,
            device,
            random_start=True,
            constraint_fn=cfn,
            batch_size=batch_size if not one_batch else None,
        )
    elif attack == "fgsm":
        from src.attacks.fgsm import fgsm_attack

        x_adv_t = fgsm_attack(
            model,
            x_tensor,
            y_tensor,
            epsilon,
            device,
            constraint_fn=cfn,
            chunk_size=batch_size,
        )
        meta_out["pgd_alpha"] = None
        meta_out["pgd_steps"] = None
    else:
        raise ValueError(f"Unsupported attack {attack!r}; use pgd or fgsm")

    return x_adv_t.cpu().numpy(), x_sub, meta_out


def audit_from_attack_run(
    cfg: dict,
    *,
    run_id: str,
    attack: str,
    attack_mode: str,
    epsilon: float,
    n_samples: int,
    one_batch: bool,
    seed: int,
) -> dict:
    results_dir = Path(cfg["paths"]["results_dir"])
    run_dir = results_dir / "attacks" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    checkpoint = manifest.get("checkpoint")
    x_test, y_test, feature_names, scaler, groups = _load_test_batch(cfg, None, seed)

    x_adv_path = discover_x_adv_file(run_dir, attack, attack_mode, epsilon)
    source = "loaded"
    gen_meta: dict = {}

    if x_adv_path is not None:
        x_adv = load_x_adv_array(x_adv_path)
        if len(x_adv) != len(x_test):
            n = min(len(x_adv), len(x_test))
            x_adv = x_adv[:n]
            x_test = x_test[:n]
            y_test = y_test[:n]
    else:
        sim_n = cfg["attacks"].get("batch_size", 256) if one_batch else n_samples
        x_adv, x_test, gen_meta = generate_constrained_adv(
            cfg,
            checkpoint=checkpoint,
            x_test=x_test,
            y_test=y_test,
            attack=attack,
            attack_mode=attack_mode,
            epsilon=epsilon,
            n_samples=sim_n,
            one_batch=one_batch,
        )
        source = "simulated_pgd" if attack == "pgd" else "simulated_fgsm"

    report = audit_scaled_adv_batch(
        x_adv,
        x_test,
        scaler=scaler,
        feature_names=feature_names,
        groups=groups,
        cfg=cfg,
    )
    report.update(
        {
            "audit_kind": "attack_run",
            "attack_run_id": run_id,
            "attack": attack,
            "attack_mode": attack_mode,
            "epsilon": epsilon,
            "x_adv_source": source,
            "x_adv_path": str(x_adv_path) if x_adv_path else None,
            "manifest_n_test_samples": manifest.get("n_test_samples"),
            "checkpoint": manifest.get("checkpoint"),
            "baseline_run_id": manifest.get("baseline_run_id"),
            "test_subsample_seed": seed,
            "test_subsample_cap": cfg["attacks"].get("max_test_samples"),
        }
    )
    if gen_meta:
        gen_meta.setdefault("baseline_run_id", manifest.get("baseline_run_id"))
        report["generation"] = gen_meta
    report["feature_groups"] = {k: len(v) for k, v in groups.items()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw constraint violations.")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--from-attack-run",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Audit adversarial outputs under results/attacks/<RUN_ID>/",
    )
    parser.add_argument(
        "--attack",
        choices=("pgd", "fgsm"),
        default="pgd",
        help="Attack type when using --from-attack-run (default: pgd)",
    )
    parser.add_argument(
        "--attack-mode",
        choices=("constrained", "unconstrained"),
        default="constrained",
        help="Constraint mode for attack replay (default: constrained)",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="When simulating attacks, run only one PGD batch (batch_size rows)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.from_attack_run:
        report = audit_from_attack_run(
            cfg,
            run_id=args.from_attack_run,
            attack=args.attack,
            attack_mode=args.attack_mode,
            epsilon=args.epsilon,
            n_samples=args.n_samples,
            one_batch=args.one_batch,
            seed=args.seed,
        )
        out = (
            args.out
            or Path(cfg["paths"]["results_dir"])
            / "attacks"
            / args.from_attack_run
            / f"constraint_audit_{args.attack}_{args.attack_mode}_{_epsilon_tag(args.epsilon)}.json"
        )
    else:
        x_test, _, feature_names, scaler, groups = _load_test_batch(
            cfg, args.n_samples, args.seed
        )
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
