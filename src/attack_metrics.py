from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


def attack_success_rate(
    y_true: np.ndarray,
    y_clean_pred: np.ndarray,
    y_adv_pred: np.ndarray,
) -> float:
    """Fraction of clean-correct samples that are misclassified under attack."""
    y_true = np.asarray(y_true)
    y_clean_pred = np.asarray(y_clean_pred)
    y_adv_pred = np.asarray(y_adv_pred)

    clean_correct = y_clean_pred == y_true
    n_clean_correct = int(clean_correct.sum())
    if n_clean_correct == 0:
        return 0.0

    flipped = clean_correct & (y_adv_pred != y_true)
    return float(flipped.sum() / n_clean_correct)


def robust_accuracy(y_true: np.ndarray, y_adv_pred: np.ndarray) -> float:
    """Accuracy on adversarial predictions."""
    return float(accuracy_score(y_true, y_adv_pred))


def _resolve_attack_predictions(
    model_or_preds: Any,
    y_clean: np.ndarray,
    y_adv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model_or_preds, (tuple, list)) and len(model_or_preds) == 2:
        return np.asarray(model_or_preds[0]), np.asarray(model_or_preds[1])

    if isinstance(model_or_preds, dict):
        return (
            np.asarray(model_or_preds["y_clean_pred"]),
            np.asarray(model_or_preds["y_adv_pred"]),
        )

    predict = getattr(model_or_preds, "predict", None)
    if callable(predict):
        return np.asarray(predict(y_clean)), np.asarray(predict(y_adv))

    if callable(model_or_preds):
        return np.asarray(model_or_preds(y_clean)), np.asarray(model_or_preds(y_adv))

    raise TypeError(
        "model_or_preds must be (y_clean_pred, y_adv_pred), a dict with "
        "'y_clean_pred'/'y_adv_pred', or an object with .predict()"
    )


def evaluate_attack(
    model_or_preds: Any,
    y_true: np.ndarray,
    y_clean: np.ndarray,
    y_adv: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Compute clean vs. adversarial metrics for attack evaluation."""
    y_true = np.asarray(y_true)
    y_clean_pred, y_adv_pred = _resolve_attack_predictions(
        model_or_preds, y_clean, y_adv
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_clean_pred)),
        "macro_f1": float(
            f1_score(y_true, y_adv_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_adv_pred, average="weighted", zero_division=0)
        ),
        "asr": attack_success_rate(y_true, y_clean_pred, y_adv_pred),
        "robust_accuracy": robust_accuracy(y_true, y_adv_pred),
        "per_class_recall": {},
    }

    label_ids = np.unique(np.concatenate([y_true, y_clean_pred, y_adv_pred]))
    if len(label_names) == len(label_ids) and set(label_ids) == set(range(len(label_names))):
        target_names = label_names
        report_labels = label_ids
    else:
        id_to_name = {i: label_names[i] for i in range(len(label_names))}
        target_names = [id_to_name.get(int(i), f"class_{i}") for i in label_ids]
        report_labels = label_ids

    report = classification_report(
        y_true,
        y_adv_pred,
        labels=report_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    for name in target_names:
        if name in report:
            metrics["per_class_recall"][name] = float(report[name]["recall"])

    return metrics


def save_attack_results(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
