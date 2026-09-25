"""Scale initial noise in the model's normalized latent space."""

import math

import torch


def contract_noise(noise: torch.Tensor, strength: float) -> torch.Tensor:
    """1 leaves initial noise unchanged; 0 removes it.

    This does not change later restart or ancestral noise. The preset value
    is 0.70; lower values are available for experimentation.
    """
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("contraction must be finite and between 0 and 1")
    return noise if strength == 1.0 else noise * strength
