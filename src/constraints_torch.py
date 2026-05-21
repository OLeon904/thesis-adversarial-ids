"""
Torch wrappers for physical-feasibility projection (PGD / FGSM inner loops).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from src.constraints import (
    ConstraintMode,
    _FLOW_DURATION,
    _FLOW_IAT_MAX,
    _FLOW_IAT_MEAN,
    _EPS_FLOOR,
    _RELAXED_COHERENCE_BLEND,
    build_index_masks,
    load_feature_groups,
)

__all__ = [
    "ConstraintMode",
    "load_feature_groups",
    "build_index_masks",
    "build_torch_index_tensors",
    "project_perturbation_torch",
    "project_batch_torch",
]


def build_torch_index_tensors(
    feature_names: list[str],
    groups: dict[str, list[str]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> dict[str, torch.Tensor | int | None]:
    """Materialize numpy index masks as torch tensors on ``device``."""
    (
        immutable_mask,
        discrete_indices,
        continuous_indices,
        iat_indices,
        duration_index,
    ) = build_index_masks(feature_names, groups)

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    dev = device if device is not None else "cpu"

    return {
        "immutable_mask": torch.as_tensor(immutable_mask, device=dev),
        "discrete_indices": torch.as_tensor(discrete_indices, device=dev, dtype=dtype),
        "continuous_indices": torch.as_tensor(continuous_indices, device=dev, dtype=dtype),
        "iat_indices": torch.as_tensor(iat_indices, device=dev, dtype=dtype),
        "duration_index": duration_index,
        "iat_mean_index": name_to_idx.get(_FLOW_IAT_MEAN),
        "iat_max_index": name_to_idx.get(_FLOW_IAT_MAX),
    }


def _apply_iat_duration_coherence_torch(
    x: torch.Tensor,
    *,
    duration_index: int | None,
    iat_mean_index: int | None,
    iat_max_index: int | None,
    mode: ConstraintMode,
    relaxed_blend: float,
) -> torch.Tensor:
    if duration_index is None:
        return x

    duration = x[..., duration_index]
    lower = torch.zeros_like(duration)
    if iat_max_index is not None:
        lower = torch.maximum(lower, x[..., iat_max_index])
    if iat_mean_index is not None:
        lower = torch.maximum(lower, x[..., iat_mean_index])

    if mode == "strict":
        x = x.clone()
        x[..., duration_index] = torch.maximum(duration, lower)
        return x

    deficit = lower - duration
    x = x.clone()
    x[..., duration_index] = duration + relaxed_blend * torch.clamp(deficit, min=0.0)
    return x


def project_batch_torch(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    groups: dict[str, list[str]],
    *,
    feature_names: list[str],
    mode: ConstraintMode = "strict",
    tensors: dict[str, torch.Tensor | int | None] | None = None,
    max_perturbation_ratio: float = 0.20,
    relaxed_coherence_blend: float = _RELAXED_COHERENCE_BLEND,
) -> torch.Tensor:
    """Batch projection for torch tensors, shape (n_samples, n_features)."""
    if x_adv.shape != x_orig.shape:
        raise ValueError(f"shape mismatch: {x_adv.shape} vs {x_orig.shape}")
    if x_adv.dim() != 2:
        raise ValueError(f"expected 2D batch, got shape {tuple(x_adv.shape)}")

    if tensors is None:
        tensors = build_torch_index_tensors(
            feature_names,
            groups,
            device=x_adv.device,
        )

    immutable_mask = tensors["immutable_mask"]
    continuous_indices = tensors["continuous_indices"]
    discrete_idx = tensors["discrete_indices"]
    duration_index = tensors["duration_index"]
    iat_mean_index = tensors["iat_mean_index"]
    iat_max_index = tensors["iat_max_index"]
    assert isinstance(continuous_indices, torch.Tensor)
    assert isinstance(discrete_idx, torch.Tensor)
    assert isinstance(immutable_mask, torch.Tensor)

    x = x_adv.clone()

    if immutable_mask.any():
        x[:, immutable_mask] = x_orig[:, immutable_mask]

    if continuous_indices.numel() > 0:
        orig_c = x_orig[:, continuous_indices]
        bound = max_perturbation_ratio * torch.maximum(
            orig_c.abs(),
            torch.tensor(_EPS_FLOOR, device=x.device, dtype=x.dtype),
        )
        x[:, continuous_indices] = torch.clamp(
            x[:, continuous_indices],
            orig_c - bound,
            orig_c + bound,
        )

    if discrete_idx.numel() > 0:
        x[:, discrete_idx] = torch.clamp(torch.round(x[:, discrete_idx]), min=0.0)

    dur_idx = duration_index if isinstance(duration_index, int) else None
    mean_idx = iat_mean_index if isinstance(iat_mean_index, int) else None
    max_idx = iat_max_index if isinstance(iat_max_index, int) else None

    return _apply_iat_duration_coherence_torch(
        x,
        duration_index=dur_idx,
        iat_mean_index=mean_idx,
        iat_max_index=max_idx,
        mode=mode,
        relaxed_blend=relaxed_coherence_blend,
    )


def project_perturbation_torch(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    groups: dict[str, list[str]],
    mode: ConstraintMode = "strict",
    *,
    feature_names: list[str] | None = None,
    tensors: dict[str, torch.Tensor | int | None] | None = None,
    max_perturbation_ratio: float = 0.20,
    relaxed_coherence_blend: float = _RELAXED_COHERENCE_BLEND,
) -> torch.Tensor:
    """Single-sample torch projection (1D or (1, n_features))."""
    squeezed = False
    if x_adv.dim() == 1:
        x_adv = x_adv.unsqueeze(0)
        x_orig = x_orig.unsqueeze(0)
        squeezed = True

    if feature_names is None:
        feature_names = [str(i) for i in range(x_adv.shape[-1])]

    out = project_batch_torch(
        x_adv,
        x_orig,
        groups,
        feature_names=feature_names,
        mode=mode,
        tensors=tensors,
        max_perturbation_ratio=max_perturbation_ratio,
        relaxed_coherence_blend=relaxed_coherence_blend,
    )
    return out.squeeze(0) if squeezed else out


def load_torch_constraint_bundle(
    metadata_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[dict[str, list[str]], list[str], dict[str, torch.Tensor | int | None]]:
    """Load groups, feature names, and pre-built torch index tensors."""
    groups = load_feature_groups(metadata_path)
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        import json

        meta = json.load(f)
    feature_names: list[str] = meta["feature_names"]
    tensors = build_torch_index_tensors(feature_names, groups, device=device)
    return groups, feature_names, tensors


def make_constraint_fn(
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
    cfg: dict,
    *,
    scaler: "StandardScaler | None" = None,
) -> "Callable[[torch.Tensor, torch.Tensor], torch.Tensor]":
    """
    Build PGD/FGSM hook: ``constraint_fn(x_adv, x_orig) -> x_adv``.

    When *scaler* is provided (required for thesis experiments), projection runs
    in raw physical feature space via inverse_transform → project_batch → transform.
    Without *scaler*, falls back to scaled-space projection (legacy; not valid for
    physical-feasibility claims).
    """
    from typing import Callable

    from sklearn.preprocessing import StandardScaler

    if scaler is not None:
        from src.raw_space_constraints import make_raw_space_constraint_fn

        return make_raw_space_constraint_fn(
            scaler, feature_names, feature_groups, cfg
        )

    ccfg = cfg.get("constraints", {})
    mode: ConstraintMode = ccfg.get("mode", "relaxed")
    max_ratio = float(ccfg.get("max_perturbation_ratio", 0.20))

    def constraint_fn_scaled(x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        return project_batch_torch(
            x_adv,
            x_orig,
            feature_groups,
            feature_names=feature_names,
            mode=mode,
            max_perturbation_ratio=max_ratio,
        )

    return constraint_fn_scaled
