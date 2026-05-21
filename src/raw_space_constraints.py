"""
Project physical-feasibility constraints in raw CICFlowMeter feature space.

The MLP is trained on StandardScaler-normalized inputs. Gradients and L_inf steps
run in scaled space; constrained attacks inverse-transform to raw space, apply
``project_batch`` (immutable mask, 20% caps, integer counts, IAT–duration coherence
in physical units), then transform back for model forward passes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.constraints import ConstraintMode, build_index_masks, project_batch

ConstraintFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class RawSpaceConstraintProjector:
    """Torch hook: project constraints on inverse-scaled (physical) features."""

    def __init__(
        self,
        scaler: StandardScaler,
        feature_names: list[str],
        feature_groups: dict[str, list[str]],
        *,
        mode: ConstraintMode = "relaxed",
        max_perturbation_ratio: float = 0.20,
        relaxed_coherence_blend: float = 0.5,
    ) -> None:
        self.scaler = scaler
        self.feature_names = feature_names
        self.feature_groups = feature_groups
        self.mode = mode
        self.max_perturbation_ratio = max_perturbation_ratio
        self.relaxed_coherence_blend = relaxed_coherence_blend
        (
            self._immutable_mask,
            self._discrete_indices,
            self._continuous_indices,
            self._iat_indices,
            self._duration_index,
        ) = build_index_masks(feature_names, feature_groups)

    def project_numpy(
        self,
        x_adv_scaled: np.ndarray,
        x_orig_scaled: np.ndarray,
    ) -> np.ndarray:
        """Batch or single-sample numpy API (scaled in → scaled out)."""
        single = x_adv_scaled.ndim == 1
        xa = np.atleast_2d(np.asarray(x_adv_scaled, dtype=np.float64))
        xo = np.atleast_2d(np.asarray(x_orig_scaled, dtype=np.float64))
        xa_raw = self.scaler.inverse_transform(xa)
        xo_raw = self.scaler.inverse_transform(xo)
        projected_raw = project_batch(
            xa_raw,
            xo_raw,
            self.feature_groups,
            feature_names=self.feature_names,
            mode=self.mode,
            immutable_mask=self._immutable_mask,
            discrete_indices=self._discrete_indices,
            continuous_indices=self._continuous_indices,
            iat_indices=self._iat_indices,
            duration_index=self._duration_index,
            max_perturbation_ratio=self.max_perturbation_ratio,
            relaxed_coherence_blend=self.relaxed_coherence_blend,
        )
        out = self.scaler.transform(projected_raw).astype(np.float32)
        return out[0] if single else out

    def __call__(
        self,
        x_adv: torch.Tensor,
        x_orig: torch.Tensor,
    ) -> torch.Tensor:
        device = x_adv.device
        dtype = x_adv.dtype
        xa = x_adv.detach().cpu().numpy()
        xo = x_orig.detach().cpu().numpy()
        projected = self.project_numpy(xa, xo)
        return torch.from_numpy(projected).to(device=device, dtype=dtype)


def make_raw_space_constraint_fn(
    scaler: StandardScaler,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
    cfg: dict[str, Any],
) -> ConstraintFn:
    """Build PGD/FGSM constraint hook using physical (raw) feature space."""
    ccfg = cfg.get("constraints", {})
    mode: ConstraintMode = ccfg.get("mode", "relaxed")
    max_ratio = float(ccfg.get("max_perturbation_ratio", 0.20))
    projector = RawSpaceConstraintProjector(
        scaler,
        feature_names,
        feature_groups,
        mode=mode,
        max_perturbation_ratio=max_ratio,
    )
    return projector
