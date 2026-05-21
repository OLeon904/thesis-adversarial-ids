from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mlp import MLPClassifier

ConstraintFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _to_tensor(
    x: torch.Tensor | np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)
    return x.to(device=device, dtype=torch.float32)


def _to_label_tensor(
    y: torch.Tensor | np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(y, np.ndarray):
        return torch.from_numpy(y).to(device=device, dtype=torch.long)
    return y.to(device=device, dtype=torch.long)


def project_linf_ball(
    x_adv: torch.Tensor,
    x_orig: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Project adversarial points back into the L_inf epsilon ball around x_orig."""
    delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
    return x_orig + delta


def _pgd_attack_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool,
    constraint_fn: ConstraintFn | None,
) -> torch.Tensor:
    """Run PGD-L_inf on a single batch that fits in memory."""
    was_training = model.training
    model.eval()

    x_orig = x.detach()
    x_adv = x_orig.clone()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = project_linf_ball(x_adv, x_orig, epsilon)
        if constraint_fn is not None:
            x_adv = constraint_fn(x_adv, x_orig)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = project_linf_ball(x_adv, x_orig, epsilon)
            if constraint_fn is not None:
                x_adv = constraint_fn(x_adv, x_orig)

    if was_training:
        model.train()
    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    epsilon: float,
    alpha: float,
    steps: int,
    device: torch.device,
    random_start: bool = True,
    constraint_fn: ConstraintFn | None = None,
    batch_size: int | None = 256,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) under L_inf threat model.

    Untargeted attack: maximizes cross-entropy loss on true labels. Each step
    takes a signed gradient step, projects back to the L_inf epsilon ball, then
    optionally applies ``constraint_fn`` for constrained PGD (e.g. feature masks,
    integer projection, timing coherence).

    Processes inputs in chunks when ``batch_size`` is set to limit memory use.
    """
    x_t = _to_tensor(x, device)
    y_t = _to_label_tensor(y, device)

    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError(
            f"x and y batch sizes must match, got {x_t.shape[0]} and {y_t.shape[0]}"
        )
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if epsilon < 0 or alpha <= 0:
        raise ValueError(f"epsilon must be >= 0 and alpha must be > 0")

    n = x_t.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        return _pgd_attack_batch(
            model,
            x_t,
            y_t,
            epsilon,
            alpha,
            steps,
            random_start,
            constraint_fn,
        )

    adv_chunks: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = _pgd_attack_batch(
            model,
            x_t[start:end],
            y_t[start:end],
            epsilon,
            alpha,
            steps,
            random_start,
            constraint_fn,
        )
        adv_chunks.append(chunk)
    return torch.cat(adv_chunks, dim=0)


def pgd_attack_mlp(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    steps: int,
    device: torch.device,
    random_start: bool = True,
    constraint_fn: ConstraintFn | None = None,
    batch_size: int | None = 256,
) -> np.ndarray:
    """
    NumPy-facing PGD wrapper for ``MLPClassifier``.

    Returns adversarial examples with the same shape as ``X``.
    """
    x_adv = pgd_attack(
        model,
        X,
        y,
        epsilon,
        alpha,
        steps,
        device,
        random_start=random_start,
        constraint_fn=constraint_fn,
        batch_size=batch_size,
    )
    return x_adv.cpu().numpy()


def build_attack_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract PGD hyperparameters from project config."""
    acfg = cfg["attacks"]
    pgd_cfg = acfg["pgd"]
    return {
        "epsilon_values": acfg["epsilon_values"],
        "alpha": pgd_cfg["alpha"],
        "steps": pgd_cfg["steps"],
        "steps_eval": pgd_cfg["steps_eval"],
        "batch_size": acfg["batch_size"],
        "max_test_samples": acfg.get("max_test_samples"),
    }
