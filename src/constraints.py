"""
Physical-feasibility projection for feature-space adversarial examples.

Enforces immutable masking, bounded continuous perturbation (literature-style
20% cap relative to each original sample), integer count features, and
Flow Duration vs Flow IAT coherence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from src.features import classify_columns

ConstraintMode = Literal["strict", "relaxed"]

# Coherence uses aggregate flow IAT statistics (not Fwd/Bwd-only columns).
_FLOW_IAT_MEAN = "Flow IAT Mean"
_FLOW_IAT_MAX = "Flow IAT Max"
_FLOW_DURATION = "Flow Duration"

# Avoid zero-width ratio bands on near-zero originals. Use with raw (physical) features.
_EPS_FLOOR = 1e-8

# Relaxed mode: fraction of coherence deficit applied per projection step.
_RELAXED_COHERENCE_BLEND = 0.5


def load_feature_groups(metadata_path: str | Path) -> dict[str, list[str]]:
    """Load feature names from metadata.json and classify constraint groups."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names: list[str] = meta["feature_names"]
    return classify_columns(feature_names)


def build_index_masks(
    feature_names: list[str],
    groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]:
    """
    Build index structures for vectorized projection.

    Returns
    -------
    immutable_mask : bool ndarray, shape (n_features,)
    discrete_indices : int ndarray
    continuous_indices : int ndarray
    iat_indices : int ndarray
    duration_index : int or None
    """
    n_features = len(feature_names)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    immutable_mask = np.zeros(n_features, dtype=bool)
    for name in groups.get("immutable", []):
        idx = name_to_idx.get(name)
        if idx is not None:
            immutable_mask[idx] = True

    def _indices(names: list[str]) -> np.ndarray:
        return np.array(
            [name_to_idx[n] for n in names if n in name_to_idx],
            dtype=np.int64,
        )

    discrete_indices = _indices(groups.get("discrete_integer", []))
    continuous_indices = _indices(groups.get("continuous", []))
    iat_indices = _indices(groups.get("iat", []))

    duration_index: int | None = name_to_idx.get(_FLOW_DURATION)

    return (
        immutable_mask,
        discrete_indices,
        continuous_indices,
        iat_indices,
        duration_index,
    )


def _coherence_indices(feature_names: list[str]) -> tuple[int | None, int | None]:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    return (
        name_to_idx.get(_FLOW_IAT_MEAN),
        name_to_idx.get(_FLOW_IAT_MAX),
    )


def _apply_iat_duration_coherence(
    x: np.ndarray,
    *,
    duration_index: int | None,
    iat_mean_index: int | None,
    iat_max_index: int | None,
    mode: ConstraintMode,
    relaxed_blend: float = _RELAXED_COHERENCE_BLEND,
) -> np.ndarray:
    """Ensure Flow Duration >= Flow IAT Max and Flow IAT Mean."""
    if duration_index is None:
        return x

    duration = x[..., duration_index]
    lower = np.zeros_like(duration)
    if iat_max_index is not None:
        lower = np.maximum(lower, x[..., iat_max_index])
    if iat_mean_index is not None:
        lower = np.maximum(lower, x[..., iat_mean_index])

    if mode == "strict":
        x[..., duration_index] = np.maximum(duration, lower)
        return x

    deficit = lower - duration
    x[..., duration_index] = duration + relaxed_blend * np.maximum(deficit, 0.0)
    return x


def project_perturbation(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    groups: dict[str, list[str]],
    mode: ConstraintMode = "strict",
    *,
    feature_names: list[str] | None = None,
    immutable_mask: np.ndarray | None = None,
    discrete_indices: np.ndarray | None = None,
    continuous_indices: np.ndarray | None = None,
    iat_indices: np.ndarray | None = None,
    duration_index: int | None = None,
    max_perturbation_ratio: float = 0.20,
    relaxed_coherence_blend: float = _RELAXED_COHERENCE_BLEND,
) -> np.ndarray:
    """
    Project a single adversarial feature vector onto the feasible set.

    Steps: zero immutable delta, bound continuous features, round discrete
    counts, then IAT–duration coherence (strict hard max vs relaxed soft lift).
    """
    x_adv = np.asarray(x_adv, dtype=np.float64)
    x_orig = np.asarray(x_orig, dtype=np.float64)
    if x_adv.shape != x_orig.shape:
        raise ValueError(f"shape mismatch: x_adv {x_adv.shape} vs x_orig {x_orig.shape}")

    if feature_names is None:
        n = x_adv.shape[-1]
        feature_names = [str(i) for i in range(n)]

    if immutable_mask is None:
        (
            immutable_mask,
            discrete_indices,
            continuous_indices,
            iat_indices,
            duration_index,
        ) = build_index_masks(feature_names, groups)

    out = project_batch(
        x_adv[np.newaxis, :],
        x_orig[np.newaxis, :],
        groups,
        feature_names=feature_names,
        mode=mode,
        immutable_mask=immutable_mask,
        discrete_indices=discrete_indices,
        continuous_indices=continuous_indices,
        iat_indices=iat_indices,
        duration_index=duration_index,
        max_perturbation_ratio=max_perturbation_ratio,
        relaxed_coherence_blend=relaxed_coherence_blend,
    )
    return out[0]


def project_batch(
    x_adv: np.ndarray,
    x_orig: np.ndarray,
    groups: dict[str, list[str]],
    *,
    feature_names: list[str],
    mode: ConstraintMode = "strict",
    immutable_mask: np.ndarray | None = None,
    discrete_indices: np.ndarray | None = None,
    continuous_indices: np.ndarray | None = None,
    iat_indices: np.ndarray | None = None,
    duration_index: int | None = None,
    max_perturbation_ratio: float = 0.20,
    relaxed_coherence_blend: float = _RELAXED_COHERENCE_BLEND,
) -> np.ndarray:
    """
    Project a batch of adversarial examples, shape (n_samples, n_features).

    ``iat_indices`` is returned by ``build_index_masks`` for PGD masking;
    coherence projection uses Flow IAT Mean / Max only.
    """
    x_adv = np.asarray(x_adv, dtype=np.float64)
    x_orig = np.asarray(x_orig, dtype=np.float64)
    if x_adv.shape != x_orig.shape:
        raise ValueError(f"shape mismatch: {x_adv.shape} vs {x_orig.shape}")
    if x_adv.ndim != 2:
        raise ValueError(f"expected 2D batch, got shape {x_adv.shape}")

    if immutable_mask is None:
        (
            immutable_mask,
            discrete_indices,
            continuous_indices,
            iat_indices,
            duration_index,
        ) = build_index_masks(feature_names, groups)

    x = x_adv.copy()

    # 1) Immutable: zero perturbation on locked header/port/flag fields.
    if immutable_mask.any():
        x[:, immutable_mask] = x_orig[:, immutable_mask]

    # 2) Continuous: per-sample ratio cap relative to original (REF-028 style).
    if continuous_indices.size > 0:
        orig_c = x_orig[:, continuous_indices]
        bound = max_perturbation_ratio * np.maximum(np.abs(orig_c), _EPS_FLOOR)
        x[:, continuous_indices] = np.clip(
            x[:, continuous_indices],
            orig_c - bound,
            orig_c + bound,
        )

    # 3) Discrete integer counts: round and enforce non-negativity.
    if discrete_indices.size > 0:
        x[:, discrete_indices] = np.maximum(np.rint(x[:, discrete_indices]), 0.0)

    # 4) IAT–duration coherence on projected IAT values.
    iat_mean_index, iat_max_index = _coherence_indices(feature_names)
    x = _apply_iat_duration_coherence(
        x,
        duration_index=duration_index,
        iat_mean_index=iat_mean_index,
        iat_max_index=iat_max_index,
        mode=mode,
        relaxed_blend=relaxed_coherence_blend,
    )

    return x.astype(x_adv.dtype, copy=False)


def build_masks_from_metadata(
    metadata_path: str | Path,
) -> tuple[dict[str, list[str]], list[str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]]:
    """Convenience: load groups, names, and index masks from metadata.json."""
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names: list[str] = meta["feature_names"]
    groups = classify_columns(feature_names)
    masks = build_index_masks(feature_names, groups)
    return groups, feature_names, masks
