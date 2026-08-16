"""Pure settings tests; no ComfyUI installation required."""

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

# Stub ComfyUI-dependent engine modules; resolve_settings does not call them.
sampling = types.ModuleType("cyberkrea_sampler.sampling")
sampling.run_sampling = None
sys.modules[sampling.__name__] = sampling
schedules = types.ModuleType("cyberkrea_sampler.schedules")
schedules.build_schedule = None
sys.modules[schedules.__name__] = schedules
nodes = load_module("nodes")


def test_preset_defaults():
    balanced = nodes.resolve_settings(
        "balanced", 12, "euler", 0.25, 0.65, True, 0.60, 1.0, 0.10, 0.70
    )
    assert balanced["steps"] == 12
    assert balanced["detail"] == 0.60
    assert balanced["sampler"] == "euler"


def test_overrides():
    result = nodes.resolve_settings(
        "balanced",
        steps=10,
        sampler="euler_2m",
        restart_frac=0.15,
        sigma_r=0.55,
        plunge=False,
        detail=0.42,
        eta0=0.50,
        sigma_gate=0.20,
        contraction=0.85,
    )
    assert result["steps"] == 10
    assert result["detail"] == 0.42
    assert result["sampler"] == "euler_2m"
    assert result["order"] == 2
    assert result["restart_frac"] == 0.15
    assert result["sigma_r"] == 0.55
    assert result["plunge"] is False
    assert result["eta0"] == 0.50
    assert result["sigma_gate"] == 0.20
    assert result["contraction"] == 0.85


if __name__ == "__main__":
    test_preset_defaults()
    test_overrides()
    print("CyberKrea Sampler settings tests passed")
