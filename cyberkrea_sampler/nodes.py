"""Compact CyberKrea sampler node for Krea 2 Turbo.

The sampling engine is derived from ComfyUI-KreaPhoton.  This node deliberately
exposes one MODEL path only, which keeps LoRA and NegPiP model patches intact.
"""

import contextlib
import math

from .presets import ALPHA, DEFAULT_PRESET, GUIDANCE, PRESETS
from .sampling import run_sampling
from .schedules import build_schedule


CATEGORY = "CyberKrea"
PREVIEW_METHODS = ["default", "auto", "latent2rgb", "taesd", "none"]
PRESET_NAMES = list(PRESETS)
SAMPLER_NAMES = ["euler", "euler_2m"]

_ORDER_FROM_SAMPLER_NAME = {"euler": 1, "euler_2m": 2}
_DEFAULTS = PRESETS[DEFAULT_PRESET]
_FLOAT_RANGES = {
    "restart_frac": (0.0, 0.60), "sigma_r": (0.0, 1.0),
    "detail": (0.0, 1.0), "eta0": (0.0, 2.0),
    "sigma_gate": (0.0, 1.0), "contraction": (0.0, 1.0),
}

_PREVIEW_TOOLTIP = (
    "Live preview during sampling. default follows ComfyUI's global setting; "
    "auto uses ComfyUI's automatic selection. taesd requires lighttaew2_1 "
    "in models/vae_approx and falls back to latent2rgb."
)
_NEGATIVE_TOOLTIP = (
    "Optional. Leave disconnected when NegPiP already handles the negative prompt. "
    "When connected, CFG is 2.25 above sigma 0.9, ramps down to 1.0 at "
    "sigma 0.7, and stays at 1.0 below it."
)
_VAE_TOOLTIP = (
    "Optional. Fully decodes only the first image for a thumbnail on this node. "
    "The LATENT output is unchanged."
)


def resolve_settings(preset, steps=None, sampler=None, restart_frac=None,
                     sigma_r=None, plunge=None, detail=None, eta0=None,
                     sigma_gate=None, contraction=None):
    """Fill omitted values from the preset; explicit controls always win."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown CyberKrea preset: {preset}")
    overrides = {
        "steps": steps,
        "sampler": sampler,
        "restart_frac": restart_frac, "sigma_r": sigma_r, "plunge": plunge,
        "detail": detail, "eta0": eta0, "sigma_gate": sigma_gate,
        "contraction": contraction,
    }
    settings = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    steps = settings["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 64:
        raise ValueError("steps must be an integer between 1 and 64")
    if settings["sampler"] not in _ORDER_FROM_SAMPLER_NAME:
        raise ValueError(f"Unknown sampler: {settings['sampler']}")
    if not isinstance(settings["plunge"], bool):
        raise ValueError("plunge must be a boolean")
    for name, (minimum, maximum) in _FLOAT_RANGES.items():
        try:
            value = float(settings[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
        settings[name] = value
    settings["order"] = _ORDER_FROM_SAMPLER_NAME[settings["sampler"]]
    settings["alpha"] = ALPHA
    return settings


@contextlib.contextmanager
def _live_preview(method):
    if method not in PREVIEW_METHODS:
        raise ValueError(f"Unknown preview method: {method}")
    if method == "default":
        yield
        return
    try:
        import latent_preview
        from comfy.cli_args import args
    except ImportError:
        yield
        return

    previous = args.preview_method
    args.preview_method = {
        "auto": latent_preview.LatentPreviewMethod.Auto,
        "latent2rgb": latent_preview.LatentPreviewMethod.Latent2RGB,
        "taesd": latent_preview.LatentPreviewMethod.TAESD,
    }.get(method, latent_preview.LatentPreviewMethod.NoPreviews)
    try:
        yield
    finally:
        args.preview_method = previous


def _result_with_preview(output, vae):
    if vae is None:
        return (output,)

    import nodes as comfy_nodes

    images = vae.decode(output["samples"][:1])
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    ui = comfy_nodes.PreviewImage().save_images(
        images, filename_prefix="CyberKrea"
    )["ui"]
    return {"ui": ui, "result": (output,)}


class CyberKreaSampler:
    """Preset-driven Krea 2 sampler with a single patched MODEL path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "preset": (PRESET_NAMES, {
                    "default": DEFAULT_PRESET,
                    "cyberkrea_presets": PRESETS,
                }),
                "steps": ("INT", {
                    "default": _DEFAULTS["steps"],
                    "min": 1,
                    "max": 64,
                    "tooltip": "Filled by the preset; may be adjusted manually.",
                }),
                "sampler": (SAMPLER_NAMES, {
                    "default": _DEFAULTS["sampler"],
                    "tooltip": "Euler or AB2 history. euler_2m needs consecutive deterministic "
                               "steps; use eta0=0 to enable it throughout each descent.",
                }),
                "restart_frac": ("FLOAT", {
                    "default": _DEFAULTS["restart_frac"],
                    "min": 0.0,
                    "max": 0.60,
                    "step": 0.01,
                    "tooltip": "Fraction of the existing step budget used by the restart. "
                               "Skipped when the budget is too small or sigma_r=0.",
                }),
                "sigma_r": ("FLOAT", {
                    "default": _DEFAULTS["sigma_r"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Restart sigma. Filled by the preset; may be adjusted manually.",
                }),
                "plunge": ("BOOLEAN", {
                    "default": _DEFAULTS["plunge"],
                    "tooltip": "Filled by the preset; may be switched manually.",
                }),
                "detail": ("FLOAT", {
                    "default": _DEFAULTS["detail"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Detail amount. Filled by the preset; may be adjusted manually.",
                }),
                "eta0": ("FLOAT", {
                    "default": _DEFAULTS["eta0"],
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Ancestral noise strength.",
                }),
                "sigma_gate": ("FLOAT", {
                    "default": _DEFAULTS["sigma_gate"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Eta is disabled at or below this sigma. Gates below 0.35 "
                               "ramp to full strength at 0.35; higher gates use a hard cutoff.",
                }),
                "contraction": ("FLOAT", {
                    "default": _DEFAULTS["contraction"],
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Scales only the initial noise: 1 leaves it unchanged, "
                               "0 removes it. Restart and ancestral noise are separate.",
                }),
                "preview_method": (PREVIEW_METHODS, {
                    "default": "default",
                    "tooltip": _PREVIEW_TOOLTIP,
                }),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": _NEGATIVE_TOOLTIP}),
                "vae": ("VAE", {"tooltip": _VAE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent_image, seed, preset,
               steps=None, sampler=None, restart_frac=None, sigma_r=None,
               plunge=None, detail=None, eta0=None, sigma_gate=None,
               contraction=None, preview_method="default",
               negative=None, vae=None):
        settings = resolve_settings(
            preset, steps, sampler, restart_frac, sigma_r, plunge,
            detail, eta0, sigma_gate, contraction,
        )
        sigmas = build_schedule(
            settings["steps"],
            alpha=settings["alpha"],
            restart_frac=settings["restart_frac"],
            sigma_r=settings["sigma_r"],
            plunge=settings["plunge"],
        )
        with _live_preview(preview_method):
            output = run_sampling(
                model,
                positive,
                negative,
                latent_image,
                sigmas,
                seed=seed,
                delta=GUIDANCE["delta"],
                lo=GUIDANCE["lo"],
                hi=GUIDANCE["hi"],
                contraction=settings["contraction"],
                detail_amount=settings["detail"],
                order=settings["order"],
                eta0=settings["eta0"],
                sigma_gate=settings["sigma_gate"],
            )
        return _result_with_preview(output, vae)


NODE_CLASS_MAPPINGS = {
    "CyberKreaSampler": CyberKreaSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberKreaSampler": "CyberKrea Sampler",
}
