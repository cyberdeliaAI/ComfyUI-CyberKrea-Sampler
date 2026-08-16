from .cyberkrea_sampler.nodes import (
    NODE_CLASS_MAPPINGS as _sampler_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _sampler_names,
)
from .cyberkrea_sampler.resolutions import (
    NODE_CLASS_MAPPINGS as _resolution_classes,
    NODE_DISPLAY_NAME_MAPPINGS as _resolution_names,
)

NODE_CLASS_MAPPINGS = {**_sampler_classes, **_resolution_classes}
NODE_DISPLAY_NAME_MAPPINGS = {**_sampler_names, **_resolution_names}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
