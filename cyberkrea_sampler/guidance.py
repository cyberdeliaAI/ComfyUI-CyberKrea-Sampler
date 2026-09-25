"""Sigma-dependent CFG applied through ComfyUI's standard guider path."""

import comfy.samplers
import torch


def smoothstep01(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def g_window(sigma: float, delta: float, lo: float = 0.7, hi: float = 0.9) -> float:
    """CFG ramp: 1 below lo, rising to 1 + delta at hi, then constant.

    The historical function name is retained. This is not a bounded window:
    guidance stays active above hi. At and below lo, exactly 1.0 allows
    ComfyUI's CFG=1 optimization when model patches have not disabled it.
    """
    if hi <= lo:
        raise ValueError("guidance hi must be greater than lo")
    return 1.0 + delta * smoothstep01((sigma - lo) / (hi - lo))


class CyberKreaGuider(comfy.samplers.CFGGuider):
    """Change CFG per evaluation while preserving model options and patches."""

    def __init__(self, model_patcher, delta: float, lo: float = 0.7, hi: float = 0.9):
        super().__init__(model_patcher)
        self.delta = delta
        self.lo = lo
        self.hi = hi

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        sigma = float(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else float(timestep)
        cond_scale = g_window(sigma, self.delta, self.lo, self.hi)
        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            self.conds.get("negative", None), self.conds.get("positive", None),
            cond_scale, model_options=model_options, seed=seed,
        )
