"""Compact CyberKrea sampler node for Krea 2 Turbo.

The sampling engine is derived from ComfyUI-KreaPhoton.  This node deliberately
exposes one MODEL path only, which keeps LoRA and NegPiP model patches intact.
"""

import contextlib

from .presets import GUIDANCE, MANIFOLD_MEAN, MANIFOLD_STD, PRESETS
from .sampling import run_sampling
from .schedules import build_schedule


CATEGORY = "CyberKrea"
PREVIEW_METHODS = ["auto", "latent2rgb", "taesd", "none"]
PRESET_NAMES = ["fast", "balanced", "quality"]
SAMPLER_NAMES = ["euler", "euler_2m"]

_PRESET_KEYS = {
    "fast": "turbo/fast",
    "balanced": "turbo/balanced",
    "quality": "turbo/quality",
}
_ORDER_FROM_SAMPLER_NAME = {"euler": 1, "euler_2m": 2}

_PREVIEW_TOOLTIP = (
    "Live preview during sampling. auto uses latent2rgb; taesd requires "
    "lighttaew2_1 in models/vae_approx and falls back to latent2rgb."
)
_NEGATIVE_TOOLTIP = (
    "Optional. Leave disconnected when NegPiP already handles the negative prompt. "
    "When connected, CyberKrea's sigma-window guidance is enabled."
)
_VAE_TOOLTIP = (
    "Optional. Connect a VAE to show a decoded thumbnail on this node. "
    "The LATENT output is unchanged."
)


def resolve_settings(preset, steps, sampler, restart_frac, sigma_r, plunge,
                     detail, eta0, sigma_gate, contraction):
    """Validate and return the visible settings selected in the node UI."""
    if preset not in _PRESET_KEYS:
        raise ValueError(f"Unknown CyberKrea preset: {preset}")
    if sampler not in _ORDER_FROM_SAMPLER_NAME:
        raise ValueError(f"Unknown sampler: {sampler}")

    return {
        "steps": int(steps),
        "sampler": sampler,
        "order": _ORDER_FROM_SAMPLER_NAME[sampler],
        "alpha": PRESETS[_PRESET_KEYS[preset]]["alpha"],
        "restart_frac": float(restart_frac),
        "sigma_r": float(sigma_r),
        "plunge": bool(plunge),
        "detail": float(detail),
        "eta0": float(eta0),
        "sigma_gate": float(sigma_gate),
        "contraction": float(contraction),
    }


@contextlib.contextmanager
def _live_preview(method):
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

    images = vae.decode(output["samples"])
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
                "preset": (PRESET_NAMES, {"default": "balanced"}),
                "steps": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Filled by the preset; may be adjusted manually.",
                }),
                "sampler": (SAMPLER_NAMES, {
                    "default": "euler",
                    "tooltip": "Filled by the preset; may be adjusted manually.",
                }),
                "restart_frac": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.60,
                    "step": 0.01,
                    "tooltip": "Fraction of steps used by the restart segment.",
                }),
                "sigma_r": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Restart sigma. Filled by the preset; may be adjusted manually.",
                }),
                "plunge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filled by the preset; may be switched manually.",
                }),
                "detail": ("FLOAT", {
                    "default": 0.60,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Detail amount. Filled by the preset; may be adjusted manually.",
                }),
                "eta0": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Ancestral noise strength.",
                }),
                "sigma_gate": ("FLOAT", {
                    "default": 0.10,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Eta is disabled below this sigma.",
                }),
                "contraction": ("FLOAT", {
                    "default": 0.70,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Initial-noise manifold contraction.",
                }),
                "preview_method": (PREVIEW_METHODS, {
                    "default": "auto",
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
               steps, sampler, restart_frac, sigma_r, plunge, detail,
               eta0, sigma_gate, contraction, preview_method="auto",
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
        guidance_mode = "off" if negative is None else (
            "window" if GUIDANCE["enabled_by_default"] else "flat"
        )

        with _live_preview(preview_method):
            output = run_sampling(
                model,
                positive,
                negative,
                latent_image,
                sigmas,
                seed=seed,
                guidance_mode=guidance_mode,
                flat_cfg=GUIDANCE["flat_cfg"],
                delta=GUIDANCE["delta"],
                lo=GUIDANCE["lo"],
                hi=GUIDANCE["hi"],
                contraction=settings["contraction"],
                per_channel_contraction=False,
                manifold_std=MANIFOLD_STD,
                manifold_mean=MANIFOLD_MEAN,
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
