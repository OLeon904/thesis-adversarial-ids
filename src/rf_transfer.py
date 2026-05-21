"""
Gradient-free transfer attack evaluation: Random Forest on MLP adversarial examples.

Random Forest is not differentiable, so we cannot run FGSM/PGD directly on the RF
surrogate. Instead, adversarial examples are crafted against the MLP (white-box
gradient attacks), then evaluated on the frozen RF without any RF gradients — a
classic *transfer attack* (Papernot et al., 2017). Success indicates that
perturbations optimized for one model evade another, which is relevant when the
deployed IDS uses tree ensembles while the attacker only has access to a neural
surrogate (or vice versa).

Public API
----------
- :func:`resolve_baseline_run` — locate ``results/baselines/<run_id>/``
- :func:`load_x_adv` — load saved ``.npy`` adversarial batch
- :func:`generate_mlp_adversarial` — FGSM/PGD on MLP (on-the-fly)
- :func:`evaluate_rf_transfer` — clean RF accuracy, transfer accuracy, ASR vs RF clean preds
- :func:`save_transfer_results` — write metrics JSON
- :func:`run_rf_transfer` — end-to-end orchestration
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from src.attack_metrics import attack_success_rate, save_attack_results
from src.io_utils import (
    find_latest_baseline_run,
    load_mlp_checkpoint,
    load_processed_data,
    load_rf_checkpoint,
)
from src.models.mlp import MLPClassifier

AttackName = Literal["fgsm", "pgd"]

__all__ = [
    "resolve_baseline_run",
    "load_x_adv",
    "generate_mlp_adversarial",
    "evaluate_rf_transfer",
    "save_transfer_results",
    "run_rf_transfer",
]


def resolve_baseline_run(cfg: dict[str, Any], run_id: str | None = None) -> Path:
    """Return baseline artifact directory (explicit *run_id* or config / latest)."""
    results_dir = Path(cfg["paths"]["results_dir"])
    if run_id:
        run_dir = results_dir / "baselines" / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Baseline run not found: {run_dir}")
        return run_dir

    configured = cfg["paths"].get("baseline_run")
    if configured:
        run_dir = results_dir / "baselines" / configured
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"Configured baseline_run={configured!r} not found: {run_dir}"
            )
        return run_dir

    return find_latest_baseline_run(results_dir)


def _subsample_test(
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, indices) for evaluation subset."""
    n = len(y_test)
    if max_samples is None or n <= max_samples:
        return X_test, y_test, np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=max_samples, replace=False))
    return X_test[indices], y_test[indices], indices


def load_x_adv(path: Path | str) -> np.ndarray:
    """Load adversarial feature matrix saved as NumPy (``.npy`` or ``.npz`` key ``x_adv``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Adversarial examples not found: {p}")

    if p.suffix == ".npz":
        with np.load(p) as data:
            if "x_adv" not in data:
                raise KeyError(f"Expected key 'x_adv' in {p}, got {list(data.files)}")
            return np.asarray(data["x_adv"])

    return np.asarray(np.load(p))


def _fgsm_batch(
    model: MLPClassifier,
    xb: torch.Tensor,
    yb: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    xb = xb.detach().clone().requires_grad_(True)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    return (xb + epsilon * xb.grad.sign()).detach()


def _pgd_batch(
    model: MLPClassifier,
    xb: torch.Tensor,
    yb: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    x_nat = xb.detach()
    x_adv = x_nat.clone()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        delta = torch.clamp(x_adv - x_nat, -epsilon, epsilon)
        x_adv = (x_nat + delta).detach()
    return x_adv


def generate_mlp_adversarial(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    attack: AttackName,
    epsilon: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Craft unconstrained L_inf adversarial examples against the MLP surrogate."""
    acfg = cfg["attacks"]
    batch_size = int(acfg["batch_size"])
    pgd_cfg = acfg["pgd"]

    model.eval()
    chunks: list[np.ndarray] = []

    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        xb = torch.from_numpy(X[start:end]).float().to(device)
        yb = torch.from_numpy(y[start:end]).long().to(device)

        if attack == "fgsm":
            adv = _fgsm_batch(model, xb, yb, epsilon)
        elif attack == "pgd":
            steps = int(pgd_cfg.get("steps_eval", pgd_cfg["steps"]))
            adv = _pgd_batch(
                model,
                xb,
                yb,
                epsilon,
                float(pgd_cfg["alpha"]),
                steps,
            )
        else:
            raise ValueError(f"Unknown attack: {attack!r}")

        chunks.append(adv.cpu().numpy())

    return np.vstack(chunks)


def evaluate_rf_transfer(
    rf_model: Any,
    y_true: np.ndarray,
    X_clean: np.ndarray,
    x_adv: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """
    Evaluate RF under MLP-transfer adversarial examples.

    Metrics
    -------
    - ``rf_clean_accuracy``: accuracy on clean test features
    - ``rf_transfer_accuracy``: accuracy on MLP-crafted ``x_adv`` (lower ⇒ stronger transfer)
    - ``asr``: attack success rate among samples RF classified correctly on clean inputs
    """
    y_true = np.asarray(y_true)
    rf_clean_pred = np.asarray(rf_model.predict(X_clean))
    rf_adv_pred = np.asarray(rf_model.predict(x_adv))

    metrics: dict[str, Any] = {
        "rf_clean_accuracy": float(accuracy_score(y_true, rf_clean_pred)),
        "rf_transfer_accuracy": float(accuracy_score(y_true, rf_adv_pred)),
        "asr": attack_success_rate(y_true, rf_clean_pred, rf_adv_pred),
        "macro_f1_adv": float(
            f1_score(y_true, rf_adv_pred, average="macro", zero_division=0)
        ),
        "n_rf_clean_correct": int((rf_clean_pred == y_true).sum()),
        "n_samples": int(len(y_true)),
    }

    from sklearn.metrics import classification_report

    n_classes = len(label_names)
    report = classification_report(
        y_true,
        rf_adv_pred,
        labels=list(range(n_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class_recall_adv"] = {
        name: float(report[name]["recall"])
        for name in label_names
        if name in report
    }

    return metrics


def save_transfer_results(metrics: dict[str, Any], path: Path) -> None:
    """Persist transfer-evaluation metrics as JSON."""
    save_attack_results(metrics, path)


def run_rf_transfer(
    cfg: dict[str, Any],
    *,
    baseline_run: str | None = None,
    attack: AttackName = "fgsm",
    epsilon: float | None = None,
    x_adv_path: Path | str | None = None,
    pilot: bool = False,
    save_x_adv: bool = True,
) -> dict[str, Any]:
    """
    Load RF + MLP baselines, obtain ``x_adv``, evaluate transfer, write JSON results.

    Returns the metrics dict (also written under ``results/rf_transfer/<run_id>/``).
    """
    baseline_dir = resolve_baseline_run(cfg, baseline_run)
    with (baseline_dir / "summary.json").open("r", encoding="utf-8") as f:
        baseline_summary = json.load(f)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    splits, meta = load_processed_data(processed_dir)
    label_names = meta["class_names"]

    max_test = cfg["attacks"].get("max_test_samples")
    if pilot and max_test is not None:
        max_test = min(max_test, 5_000)
    elif pilot:
        max_test = 5_000

    X_eval, y_eval, eval_indices = _subsample_test(
        splits["X_test"], splits["y_test"], max_test, cfg["seed"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf_path = Path(baseline_summary["models"]["random_forest"]["model_path"])
    mlp_path = Path(baseline_summary["models"]["mlp"]["model_path"])
    rf = load_rf_checkpoint(rf_path)

    eps = epsilon if epsilon is not None else float(cfg["attacks"]["epsilon_values"][0])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if pilot:
        run_id = f"pilot_{run_id}"
    out_dir = Path(cfg["paths"]["results_dir"]) / "rf_transfer" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_used = attack
    if x_adv_path is not None:
        x_adv = load_x_adv(x_adv_path)
        if len(x_adv) != len(y_eval):
            raise ValueError(
                f"x_adv rows ({len(x_adv)}) != evaluation subset ({len(y_eval)}). "
                "Regenerate or pass matching indices."
            )
        attack_used = "loaded"
    else:
        mlp, _ = load_mlp_checkpoint(mlp_path, device)
        x_adv = generate_mlp_adversarial(
            mlp, X_eval, y_eval, device, attack, eps, cfg
        )
        if save_x_adv:
            adv_file = out_dir / f"x_adv_{attack}_eps{eps:g}.npy"
            np.save(adv_file, x_adv)
            np.save(out_dir / "eval_indices.npy", eval_indices)

    metrics = evaluate_rf_transfer(rf, y_eval, X_eval, x_adv, label_names)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "pilot": pilot,
        "baseline_run": baseline_dir.name,
        "baseline_dir": str(baseline_dir),
        "attack": attack_used,
        "epsilon": eps,
        "device": str(device),
        "n_samples": int(len(y_eval)),
        "x_adv_source": str(x_adv_path) if x_adv_path else "generated",
        **metrics,
    }

    results_path = out_dir / "transfer_results.json"
    save_transfer_results(payload, results_path)

    print(f"\nRF transfer evaluation ({attack_used}, eps={eps:g})")
    print(f"  baseline: {baseline_dir.name}")
    print(f"  n={payload['n_samples']:,}")
    print(f"  rf_clean_accuracy={metrics['rf_clean_accuracy']:.4f}")
    print(f"  rf_transfer_accuracy={metrics['rf_transfer_accuracy']:.4f}")
    print(f"  asr (vs RF clean)={metrics['asr']:.4f}")
    print(f"\nResults saved to {results_path}")

    return payload
