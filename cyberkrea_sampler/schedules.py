"""Analytic rectified-flow schedules with an optional re-noise jump."""

import math

import torch

from .presets import ALPHA


TIMESTEPS = 10000
PLUNGE_SIGMA_FLOOR = 0.75


def sigma_from_t(t: float, alpha: float) -> float:
    """Flow time shift with alpha = exp(mu)."""
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def t_from_sigma(sigma: float, alpha: float) -> float:
    return sigma / (alpha - (alpha - 1.0) * sigma)


def build_schedule(n_steps: int, alpha: float = ALPHA, restart_frac: float = 0.0,
                   sigma_r: float = 0.6, plunge: bool = False) -> torch.Tensor:
    """Start at sigma 1, end at 0, and preserve the requested sampling budget.

    A restart reserves some evaluations for a linear descent from sigma_r.
    Its ascending re-noise jump does not evaluate the model. If the budget is
    too small for both segments, or sigma_r is zero, use the whole budget for
    the first descent. A one-step schedule is always [1, 0].

    Plunge stops the first descent at 0.75 before predicting the clean latent.
    It needs two evaluations; a restart after it needs at least three in total.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1:
        raise ValueError("steps must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")
    for name, value in (("restart_frac", restart_frac), ("sigma_r", sigma_r)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")

    min_structure_steps = 2 if plunge else 1
    n_r = 0
    if restart_frac > 0.0 and sigma_r > 0.0 and n_steps > min_structure_steps:
        n_r = min(max(1, round(n_steps * restart_frac)), n_steps - min_structure_steps)
    n_m = n_steps - n_r

    sigs = []
    if plunge and n_m >= 2:
        t_lo = t_from_sigma(PLUNGE_SIGMA_FLOOR, alpha)
        for i in range(n_m):
            t = 1.0 - (1.0 - t_lo) * (i / (n_m - 1))
            sigs.append(sigma_from_t(t, alpha))
    else:
        t_lo = alpha / (alpha + (TIMESTEPS - 1.0))
        for i in range(n_m):
            t = 1.0 - (1.0 - t_lo) * (i / n_m)
            sigs.append(sigma_from_t(t, alpha))
    sigs.append(0.0)

    if n_r:
        sigs.extend(sigma_r * (1.0 - j / n_r) for j in range(n_r))
        sigs.append(0.0)

    return torch.tensor(sigs, dtype=torch.float32)
