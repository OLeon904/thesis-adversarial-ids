from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    label_names: list[str],
) -> dict[str, Any]:
    """Compute thesis evaluation metrics on held-out data."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class_recall": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    for name in label_names:
        if name in report:
            metrics["per_class_recall"][name] = float(report[name]["recall"])

    if y_proba is not None and len(label_names) > 2:
        try:
            metrics["roc_auc_macro_ovr"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError as exc:
            metrics["roc_auc_macro_ovr"] = None
            metrics["roc_auc_error"] = str(exc)
    elif y_proba is not None and len(label_names) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))

    return metrics


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
