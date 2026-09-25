"""Shared sampler defaults, also sent to the frontend through node metadata.

The sampling values originate in ComfyUI-KreaPhoton's Turbo presets. They are
starting points for manual tuning, not guarantees for every model or prompt.
"""

ALPHA = 3.158
GUIDANCE = {"delta": 1.25, "lo": 0.7, "hi": 0.9}
DEFAULT_PRESET = "balanced"

_COMMON = {
    "sampler": "euler",
    "restart_frac": 0.25,
    "sigma_r": 0.65,
    "plunge": True,
    "eta0": 1.0,
    "sigma_gate": 0.10,
    "contraction": 0.70,
}

PRESETS = {
    "fast": {**_COMMON, "steps": 8, "detail": 0.50},
    "balanced": {**_COMMON, "steps": 12, "detail": 0.60},
    # Ancestral updates clear the history used by euler_2m, so Euler already
    # describes the effective method at these defaults. Use eta0=0 to try AB2.
    "quality": {**_COMMON, "steps": 16, "detail": 0.70},
}
