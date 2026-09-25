"""Preset defaults, manual overrides, validation and frontend metadata."""

import unittest

from cyberkrea_sampler.presets import DEFAULT_PRESET, PRESETS
from support import engine_modules


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.nodes = self.modules.nodes

    def test_all_presets_resolve_without_frontend(self):
        for name, defaults in PRESETS.items():
            with self.subTest(preset=name):
                result = self.nodes.resolve_settings(name)
                self.assertEqual({k: result[k] for k in defaults}, defaults)
                self.assertEqual(result["order"], 1)

    def test_explicit_controls_override_preset_including_zero_and_false(self):
        overrides = dict(steps=10, sampler="euler_2m", restart_frac=0.0,
                         sigma_r=0.55, plunge=False, detail=0.42, eta0=0.0,
                         sigma_gate=0.20, contraction=0.0)
        for name in PRESETS:
            result = self.nodes.resolve_settings(name, **overrides)
            self.assertEqual({k: result[k] for k in overrides}, overrides)
            self.assertEqual(result["order"], 2)

    def test_settings_do_not_modify_preset(self):
        result = self.nodes.resolve_settings("balanced")
        result["steps"] = 1
        self.assertEqual(PRESETS["balanced"]["steps"], 12)

    def test_invalid_inputs_are_rejected(self):
        cases = [("steps", 0), ("steps", 65), ("steps", 1.5), ("steps", True),
                 ("sampler", "unknown"), ("plunge", "false"), ("restart_frac", .61),
                 ("sigma_r", -1), ("detail", 2), ("eta0", 3),
                 ("sigma_gate", float("nan")), ("contraction", float("inf")),
                 ("contraction", "invalid")]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.nodes.resolve_settings("balanced", **{key: value})
        with self.assertRaisesRegex(ValueError, "Unknown CyberKrea preset"):
            self.nodes.resolve_settings("raw/experimental")

    def test_ui_defaults_and_metadata_use_python_presets(self):
        inputs = self.nodes.CyberKreaSampler.INPUT_TYPES()["required"]
        self.assertEqual(inputs["preset"][1]["cyberkrea_presets"], PRESETS)
        for key, value in PRESETS[DEFAULT_PRESET].items():
            self.assertEqual(inputs[key][1]["default"], value)
        self.assertEqual(inputs["preview_method"][1]["default"], "default")


if __name__ == "__main__":
    unittest.main()
