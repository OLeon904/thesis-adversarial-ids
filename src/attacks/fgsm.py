from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mlp import MLPClassifier


def load_mlp_from_checkpoint(
    path: Path | str,
    device: torch.device,
) -> MLPClassifier:
    """Load an MLP saved by train_baselines (state_dict + architecture metadata)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MLPClassifier(
        n_features=ckpt["n_features"],
        n_classes=ckpt["n_classes"],
        hidden_dims=ckpt["hidden_dims"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _fgsm_attack_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: torch.device,
    targeted: bool = False,
    constraint_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    x = x.to(device)
    y = y.to(device).long()
    x_orig = x.detach()

    x_adv = x_orig.clone().requires_grad_(True)

    model.eval()
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    if targeted:
        delta = -epsilon * grad_sign
    else:
        delta = epsilon * grad_sign

    x_adv = torch.clamp(x_orig + delta, x_orig - epsilon, x_orig + epsilon)

    if constraint_fn is not None:
        x_adv = constraint_fn(x_adv, x_orig)

    return x_adv.detach()


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: torch.device,
    targeted: bool = False,
    constraint_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) under an L_inf ball around x.

    Processes large inputs in batches of ``chunk_size`` (default 256).
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D (n_samples, n_features), got shape {tuple(x.shape)}")
    if y.dim() != 1:
        raise ValueError(f"y must be 1D (n_samples,), got shape {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y batch sizes differ: {x.shape[0]} vs {y.shape[0]}")

    n = x.shape[0]
    if n <= chunk_size:
        return _fgsm_attack_batch(
            model, x, y, epsilon, device, targeted=targeted, constraint_fn=constraint_fn
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append(
            _fgsm_attack_batch(
                model,
                x[start:end],
                y[start:end],
                epsilon,
                device,
                targeted=targeted,
                constraint_fn=constraint_fn,
            )
        )

    return torch.cat(chunks, dim=0)
