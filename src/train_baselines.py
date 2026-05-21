from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import ensure_dirs, load_config
from src.metrics import evaluate_classifier, save_metrics
from src.models.mlp import MLPClassifier, predict_mlp, train_mlp
from src.models.rf import train_rf


def subsample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y_train) <= max_samples:
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_samples,
        stratify=y_train,
        random_state=seed,
    )
    print(f"  Training subsample: {len(y_sub):,} / {len(y_train):,} rows")
    return X_sub, y_sub


def load_processed(processed_dir: Path) -> dict:
    splits = np.load(processed_dir / "splits.npz")
    with (processed_dir / "metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "X_train": splits["X_train"],
        "X_val": splits["X_val"],
        "X_test": splits["X_test"],
        "y_train": splits["y_train"],
        "y_val": splits["y_val"],
        "y_test": splits["y_test"],
        "meta": meta,
    }


def run_baselines(cfg: dict, pilot: bool = False) -> None:
    processed_dir = Path(cfg["paths"]["processed_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if pilot:
        run_id = f"pilot_{run_id}"
    out_run = results_dir / "baselines" / run_id
    out_run.mkdir(parents=True, exist_ok=True)

    data = load_processed(processed_dir)
    label_names = data["meta"]["class_names"]
    n_classes = data["meta"]["n_classes"]

    max_train = cfg["data"].get("max_train_samples")
    if pilot:
        max_train = min(max_train or 50_000, 50_000)
    X_train, y_train = subsample_train(
        data["X_train"], data["y_train"], max_train, cfg["seed"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary: dict = {
        "run_id": run_id,
        "pilot": pilot,
        "device": str(device),
        "max_train_samples": max_train,
        "train_rows_used": int(len(y_train)),
        "test_rows": int(len(data["y_test"])),
        "models": {},
    }

    # --- Random Forest ---
    rf_path = out_run / "rf_model.joblib"
    rf = train_rf(X_train, y_train, cfg, save_path=rf_path)
    rf_pred = rf.predict(data["X_test"])
    rf_proba = rf.predict_proba(data["X_test"])
    rf_metrics = evaluate_classifier(
        data["y_test"], rf_pred, rf_proba, label_names
    )
    save_metrics(rf_metrics, out_run / "rf_test_metrics.json")
    summary["models"]["random_forest"] = {
        "metrics": {k: v for k, v in rf_metrics.items() if k != "confusion_matrix"},
        "model_path": str(rf_path),
    }
    print("\nRandom Forest — test set:")
    print(f"  accuracy={rf_metrics['accuracy']:.4f}  macro_f1={rf_metrics['macro_f1']:.4f}")

    # --- MLP ---
    mlp_path = out_run / "mlp_model.pt"
    print("\nTraining MLP...")
    mlp = train_mlp(
        X_train,
        y_train,
        data["X_val"],
        data["y_val"],
        n_classes,
        cfg,
        device,
        save_path=mlp_path,
    )
    mlp_pred, mlp_proba = predict_mlp(mlp, data["X_test"], device)
    mlp_metrics = evaluate_classifier(
        data["y_test"], mlp_pred, mlp_proba, label_names
    )
    save_metrics(mlp_metrics, out_run / "mlp_test_metrics.json")
    summary["models"]["mlp"] = {
        "metrics": {k: v for k, v in mlp_metrics.items() if k != "confusion_matrix"},
        "model_path": str(mlp_path),
    }
    print("\nMLP — test set:")
    print(f"  accuracy={mlp_metrics['accuracy']:.4f}  macro_f1={mlp_metrics['macro_f1']:.4f}")

    with (out_run / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_run}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train RF and MLP baselines on CICIDS2017")
    parser.add_argument("--pilot", action="store_true", help="Use pilot_max_rows from config")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    run_baselines(cfg, pilot=args.pilot)


if __name__ == "__main__":
    main()
