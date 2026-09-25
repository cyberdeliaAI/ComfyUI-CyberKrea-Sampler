"""Real PyTorch engine tests with a small, isolated ComfyUI boundary double."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def engine_modules():
    comfy = types.ModuleType("comfy")
    modules = {"comfy": comfy}
    for name in ("samplers", "sample", "utils", "model_management"):
        module = types.ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        modules[module.__name__] = module

    guiders = []

    class CFGGuider:
        def __init__(self, model):
            self.model_patcher = model
            self.inner_model = model
            self.conds = {}
            self.sample = Mock(side_effect=lambda noise, latent, *args, **kwargs: latent + noise)
            guiders.append(self)

        def set_cfg(self, cfg):
            self.cfg = cfg

        def set_conds(self, positive, negative):
            self.conds = {"positive": positive, "negative": negative}

    comfy.samplers.CFGGuider = CFGGuider
    comfy.samplers.KSAMPLER = Mock(side_effect=lambda fn, extra_options: types.SimpleNamespace(
        sampler_function=fn, extra_options=extra_options))
    comfy.samplers.sampling_function = Mock()
    comfy.sample.fix_empty_latent_channels = Mock(side_effect=lambda model, latent: latent)
    comfy.sample.prepare_noise = Mock(side_effect=lambda latent, seed, indices=None: torch.randn(
        latent.shape, generator=torch.Generator().manual_seed(seed), dtype=latent.dtype))
    comfy.model_management.intermediate_device = lambda: "cpu"
    comfy.model_management.intermediate_dtype = lambda: torch.float32
    comfy.utils.PROGRESS_BAR_ENABLED = False
    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = Mock(return_value=Mock())
    modules["latent_preview"] = preview
    package = types.ModuleType("_cyberkrea_test")
    package.__path__ = [str(ROOT / "cyberkrea_sampler")]
    modules[package.__name__] = package
    # Restore only our modules. Rewinding all of sys.modules also unloads lazy
    # PyTorch imports, whose native operators cannot safely register twice.
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        nodes = importlib.import_module("_cyberkrea_test.nodes")
        sampling = importlib.import_module("_cyberkrea_test.sampling")
        guidance = importlib.import_module("_cyberkrea_test.guidance")
        yield types.SimpleNamespace(nodes=nodes, sampling=sampling, guidance=guidance,
                                    comfy=comfy, preview=preview, guiders=guiders)
    finally:
        for name in list(sys.modules):
            if name.startswith("_cyberkrea_test."):
                sys.modules.pop(name, None)
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
