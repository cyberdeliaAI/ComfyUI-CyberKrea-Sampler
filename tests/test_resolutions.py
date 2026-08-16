"""Tests for CyberKrea resolution selection without a ComfyUI install."""

import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "cyberkrea_sampler"


def load_module(name):
    path = PACKAGE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"cyberkrea_sampler.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType("cyberkrea_sampler")
package.__path__ = [str(PACKAGE)]
sys.modules["cyberkrea_sampler"] = package
load_module("presets")
resolutions = load_module("resolutions")


def test_resolution_tiers_and_dimensions():
    assert list(resolutions.RESOLUTION_OPTIONS) == [
        "S (~1.0 MP)", "M (~1.4 MP)", "L (~1.7 MP)", "XL (~2.1 MP)"
    ]
    assert all(len(options) == 5 for options in resolutions.RESOLUTION_OPTIONS.values())
    assert resolutions.DEFAULT_RESOLUTION == "1088x1600 (2:3)"
    assert resolutions.resolve_dimensions(
        "L (~1.7 MP)", "1088x1600 (2:3)"
    ) == (1088, 1600)


def test_every_dimension_is_latent_safe():
    for size, options in resolutions.RESOLUTION_OPTIONS.items():
        for option in options:
            width, height = resolutions.resolve_dimensions(size, option)
            assert width % 16 == 0
            assert height % 16 == 0


def test_generate_shape_without_real_torch():
    class FakeTorch:
        @staticmethod
        def zeros(shape):
            return types.SimpleNamespace(shape=tuple(shape))

    old_torch = sys.modules.get("torch")
    sys.modules["torch"] = FakeTorch
    try:
        latent, width, height = resolutions.CyberKreaEmptyLatent().generate(
            "S (~1.0 MP)", "768x1344 (9:16)", 2
        )
    finally:
        if old_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = old_torch

    assert (width, height) == (768, 1344)
    assert latent["samples"].shape == (2, 16, 168, 96)


if __name__ == "__main__":
    test_resolution_tiers_and_dimensions()
    test_every_dimension_is_latent_safe()
    test_generate_shape_without_real_torch()
    print("CyberKrea resolution tests passed")
