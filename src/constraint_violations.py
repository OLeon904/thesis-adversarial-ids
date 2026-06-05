"""Count raw-space constraint violations before/after projection."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.constraints import _EPS_FLOOR, _FLOW_DURATION, _FLOW_IAT_MAX, _FLOW_IAT_MEAN, build_index_masks

_VIOLATION_KEYS = (
    "immutable_delta",
    "continuous_cap",
    "discrete_negative",
    "discrete_non_integer",
    "iat_duration",
)


def count_raw_violations(
    x_raw: np.ndarray,
    x_orig_raw: np.ndarray,
    *,
    feature_names: list[str],
    groups: dict[str, list[str]],
    max_perturbation_ratio: float = 0.20,
    integer_tol: float = 1e-3,
) -> dict[str, int]:
    """
    Count feature-level violations on raw vectors (shape n_samples × n_features).

    Each key counts how many (sample, feature) pairs violate that rule.
    """
    x = np.asarray(x_raw, dtype=np.float64)
    xo = np.asarray(x_orig_raw, dtype=np.float64)
    if x.shape != xo.shape or x.ndim != 2:
        raise ValueError(f"expected matching 2D arrays, got {x.shape} vs {xo.shape}")

    (
        immutable_mask,
        discrete_indices,
        continuous_indices,
        _iat_indices,
        duration_index,
    ) = build_index_masks(feature_names, groups)

    counts = {k: 0 for k in _VIOLATION_KEYS}
    n, _ = x.shape

    if immutable_mask.any():
        delta = np.abs(x[:, immutable_mask] - xo[:, immutable_mask])
        counts["immutable_delta"] = int(np.sum(delta > 1e-9))

    if continuous_indices.size > 0:
        orig_c = xo[:, continuous_indices]
        bound = max_perturbation_ratio * np.maximum(np.abs(orig_c), _EPS_FLOOR)
        low = orig_c - bound
        high = orig_c + bound
        vals = x[:, continuous_indices]
        outside = (vals < low - 1e-9) | (vals > high + 1e-9)
        counts["continuous_cap"] = int(np.sum(outside))

    if discrete_indices.size > 0:
        vals = x[:, discrete_indices]
        counts["discrete_negative"] = int(np.sum(vals < -integer_tol))
        frac = np.abs(vals - np.rint(vals))
        counts["discrete_non_integer"] = int(np.sum(frac > integer_tol))

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    dur_i = duration_index
    mean_i = name_to_idx.get(_FLOW_IAT_MEAN)
    max_i = name_to_idx.get(_FLOW_IAT_MAX)
    if dur_i is not None:
        duration = x[:, dur_i]
        lower = np.zeros(n, dtype=np.float64)
        if max_i is not None:
            lower = np.maximum(lower, x[:, max_i])
        if mean_i is not None:
            lower = np.maximum(lower, x[:, mean_i])
        counts["iat_duration"] = int(np.sum(duration + 1e-9 < lower))

    return counts


def summarize_violations(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    return {"by_rule": counts, "total_violations": total}
