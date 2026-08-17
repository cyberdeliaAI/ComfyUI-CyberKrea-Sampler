"""Krea 2 resolution selector and 16-channel empty latent node."""

from .presets import (
    DEFAULT_RESOLUTION_ASPECT,
    DEFAULT_RESOLUTION_SIZE,
    RESOLUTION_BUCKETS,
)


CATEGORY = "CyberKrea"


def _build_resolution_options():
    options = {}
    dimensions = {}
    for size, aspects in RESOLUTION_BUCKETS.items():
        size_options = []
        for aspect, (width, height) in aspects.items():
            label = f"{width}x{height} ({aspect})"
            size_options.append(label)
            dimensions[(size, label)] = (width, height)
        options[size] = size_options
    return options, dimensions


RESOLUTION_OPTIONS, _DIMENSIONS = _build_resolution_options()
ALL_RESOLUTIONS = list(dict.fromkeys(
    label for labels in RESOLUTION_OPTIONS.values() for label in labels
))

_default_width, _default_height = RESOLUTION_BUCKETS[DEFAULT_RESOLUTION_SIZE][
    DEFAULT_RESOLUTION_ASPECT
]
DEFAULT_RESOLUTION = (
    f"{_default_width}x{_default_height} "
    f"({DEFAULT_RESOLUTION_ASPECT})"
)


def resolve_dimensions(size, resolution):
    """Return dimensions, preserving the aspect after a stale UI tier change."""
    try:
        return _DIMENSIONS[(size, resolution)]
    except KeyError:
        pass

    # ComfyUI can briefly restore the size widget before it refreshes the
    # dependent resolution widget. If the stale value is one of our valid
    # resolutions, select the same aspect in the active tier instead.
    if resolution in ALL_RESOLUTIONS and size in RESOLUTION_OPTIONS:
        aspect = resolution.rpartition("(")[2].removesuffix(")")
        suffix = f"({aspect})"
        for candidate in RESOLUTION_OPTIONS[size]:
            if candidate.endswith(suffix):
                return _DIMENSIONS[(size, candidate)]

    raise ValueError(
        f"Resolution {resolution!r} does not belong to size {size!r}"
    )


class CyberKreaEmptyLatent:
    """Create a Krea 2 / Wan21-compatible 16-channel empty latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size": (list(RESOLUTION_OPTIONS.keys()), {
                    "default": DEFAULT_RESOLUTION_SIZE,
                    "tooltip": "Krea 2 resolution tier; filters the resolution list.",
                }),
                "resolution": (ALL_RESOLUTIONS, {
                    "default": DEFAULT_RESOLUTION,
                    "tooltip": "Concrete Krea 2 width, height and aspect ratio.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, size, resolution, batch_size):
        import torch

        width, height = resolve_dimensions(size, resolution)
        latent = torch.zeros([int(batch_size), 16, height // 8, width // 8])
        return ({"samples": latent}, width, height)


NODE_CLASS_MAPPINGS = {
    "CyberKreaEmptyLatent": CyberKreaEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CyberKreaEmptyLatent": "CyberKrea Empty Latent",
}
