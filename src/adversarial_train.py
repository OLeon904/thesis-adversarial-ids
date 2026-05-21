from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.pgd import pgd_attack
from src.io_utils import find_latest_baseline_run, load_mlp_checkpoint, load_processed_data
from src.models.mlp import MLPClassifier
from src.train_baselines import subsample_train

ConstraintFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _mix_clean_adversarial(
    x_clean: torch.Tensor,
    y_clean: torch.Tensor,
    x_adv: torch.Tensor,
    mix_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch with (1 - mix_ratio) clean and mix_ratio adversarial samples."""
    n = x_clean.shape[0]
    n_adv = int(round(n * mix_ratio))
    n_adv = max(0, min(n, n_adv))

    if n_adv == 0:
        return x_clean, y_clean
    if n_adv == n:
        return x_adv, y_clean

    perm = torch.randperm(n, device=x_clean.device)
    adv_idx = perm[:n_adv]
    clean_idx = perm[n_adv:]

    x_mix = torch.cat([x_clean[clean_idx], x_adv[adv_idx]], dim=0)
    y_mix = torch.cat([y_clean[clean_idx], y_clean[adv_idx]], dim=0)

    shuffle = torch.randperm(x_mix.shape[0], device=x_clean.device)
    return x_mix[shuffle], y_mix[shuffle]


def _save_mlp_checkpoint(
    model: MLPClassifier,
    path: Path,
    cfg: dict,
    n_features: int,
    n_classes: int,
    extra: dict | None = None,
) -> None:
    mcfg = cfg["models"]["mlp"]
    payload: dict = {
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "n_classes": n_classes,
        "hidden_dims": mcfg["hidden_dims"],
        "dropout": mcfg["dropout"],
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def adversarial_train_mlp(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    device: torch.device,
    constraint_fn: ConstraintFn | None = None,
    save_path: Path | None = None,
) -> MLPClassifier:
    """
    One adversarial-training pass: each batch mixes clean and PGD examples.

    Early-stops on validation macro-F1 (same patience as clean ``train_mlp``).
    """
    mcfg = cfg["models"]["mlp"]
    atcfg = cfg["adversarial_training"]
    pgd_cfg = cfg["attacks"]["pgd"]

    epsilon = float(atcfg["epsilon"])
    pgd_steps = int(atcfg["pgd_steps"])
    mix_ratio = float(atcfg["mix_ratio"])
    alpha = float(pgd_cfg["alpha"])
    attack_batch_size = int(cfg["attacks"].get("batch_size", mcfg["batch_size"]))

    n_classes = int(y_train.max()) + 1

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).long(),
        ),
        batch_size=mcfg["batch_size"],
        shuffle=True,
    )
    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).long().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mcfg["learning_rate"],
        weight_decay=mcfg["weight_decay"],
    )
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6)) / len(counts)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    best_val_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = mcfg["early_stopping_patience"]
    stale = 0

    for epoch in range(mcfg["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            x_adv = pgd_attack(
                model,
                xb,
                yb,
                epsilon=epsilon,
                alpha=alpha,
                steps=pgd_steps,
                device=device,
                random_start=True,
                constraint_fn=constraint_fn,
                batch_size=attack_batch_size,
            )
            x_mix, y_mix = _mix_clean_adversarial(xb, yb, x_adv, mix_ratio)

            optimizer.zero_grad()
            loss = criterion(model(x_mix), y_mix)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            pred = logits.argmax(dim=1).cpu().numpy()
        val_acc = float((pred == y_val).mean())
        val_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{mcfg['epochs']}  "
                f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}"
            )
        if stale >= patience:
            print(f"  Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        _save_mlp_checkpoint(
            model,
            save_path,
            cfg,
            n_features=X_train.shape[1],
            n_classes=n_classes,
            extra={
                "best_val_macro_f1": best_val_f1,
                "adversarial_training": {
                    "epsilon": epsilon,
                    "pgd_steps": pgd_steps,
                    "mix_ratio": mix_ratio,
                    "alpha": alpha,
                },
            },
        )

    return model


def run_adversarial_training(
    cfg: dict,
    passes: int | None = None,
    baseline_run: str | None = None,
    pilot: bool = False,
) -> Path:
    """Run multi-pass adversarial training; checkpoints under ``results/adv_train/<run_id>/``."""
    processed_dir = Path(cfg["paths"]["processed_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])

    splits, metadata = load_processed_data(processed_dir)
    n_classes = metadata["n_classes"]

    max_train = cfg["data"].get("max_train_samples")
    if pilot:
        max_train = min(max_train or 50_000, 50_000)
    X_train, y_train = subsample_train(
        splits["X_train"], splits["y_train"], max_train, cfg["seed"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline_run_id = baseline_run or cfg["paths"].get("baseline_run")
    if baseline_run_id:
        baseline_dir = results_dir / "baselines" / baseline_run_id
    else:
        baseline_dir = find_latest_baseline_run(results_dir)
        baseline_run_id = baseline_dir.name
    mlp_ckpt = baseline_dir / "mlp_model.pt"
    if not mlp_ckpt.is_file():
        raise FileNotFoundError(
            f"Baseline MLP not found at {mlp_ckpt}. Run scripts/run_baselines.py first."
        )

    model, ckpt_meta = load_mlp_checkpoint(mlp_ckpt, device)
    model.train()
    print(f"Loaded baseline MLP from {mlp_ckpt} (run {baseline_run_id})")

    n_passes = passes if passes is not None else int(cfg["adversarial_training"]["passes"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if pilot:
        run_id = f"pilot_{run_id}"
    out_run = results_dir / "adv_train" / run_id
    out_run.mkdir(parents=True, exist_ok=True)

    pass_metrics: list[dict] = []
    for k in range(1, n_passes + 1):
        print(f"\n=== Adversarial training pass {k}/{n_passes} ===")
        save_path = out_run / f"pass_{k}.pt"
        model = adversarial_train_mlp(
            model,
            X_train,
            y_train,
            splits["X_val"],
            splits["y_val"],
            cfg,
            device,
            constraint_fn=None,
            save_path=save_path,
        )
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        pass_metrics.append(
            {
                "pass": k,
                "checkpoint": str(save_path),
                "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
            }
        )
        print(f"  Saved {save_path}")

    summary = {
        "run_id": run_id,
        "pilot": pilot,
        "device": str(device),
        "baseline_run": baseline_run_id,
        "baseline_checkpoint": str(mlp_ckpt),
        "passes": n_passes,
        "train_rows_used": int(len(y_train)),
        "adversarial_training": cfg["adversarial_training"],
        "pass_metrics": pass_metrics,
        "n_classes": n_classes,
        "checkpoint_meta": {k: v for k, v in ckpt_meta.items() if k != "state_dict"},
    }
    summary_path = out_run / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAdversarial training complete. Results: {out_run}")
    return out_run
