"""Preview settings, first-image decode and node-to-engine wiring."""

from enum import Enum
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from support import engine_modules


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.nodes = self.modules.nodes

    def test_default_preview_preserves_global_setting(self):
        args = types.SimpleNamespace(preview_method="disabled")
        cli = types.ModuleType("comfy.cli_args")
        cli.args = args
        with patch.dict(sys.modules, {"comfy.cli_args": cli}):
            with self.nodes._live_preview("default"):
                self.assertEqual(args.preview_method, "disabled")
            self.assertEqual(args.preview_method, "disabled")

    def test_explicit_preview_override_is_restored_even_on_failure(self):
        methods = Enum("LatentPreviewMethod", "Auto Latent2RGB TAESD NoPreviews")
        self.modules.preview.LatentPreviewMethod = methods
        cli = types.ModuleType("comfy.cli_args")
        cli.args = types.SimpleNamespace(preview_method=methods.NoPreviews)
        with patch.dict(sys.modules, {"comfy.cli_args": cli}):
            for name, expected in (("auto", methods.Auto), ("latent2rgb", methods.Latent2RGB),
                                   ("taesd", methods.TAESD), ("none", methods.NoPreviews)):
                with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                    with self.nodes._live_preview(name):
                        self.assertIs(cli.args.preview_method, expected)
                        raise RuntimeError("sampling failed")
                self.assertIs(cli.args.preview_method, methods.NoPreviews)
        with self.assertRaises(ValueError):
            with self.nodes._live_preview("unknown"):
                pass

    def test_vae_decodes_only_first_image_and_keeps_full_latent(self):
        samples = torch.arange(3 * 16 * 4, dtype=torch.float32).reshape(3, 16, 1, 2, 2)
        output = {"samples": samples}
        for decoded_shape in ((1, 8, 8, 3), (1, 1, 8, 8, 3)):
            vae = types.SimpleNamespace(decode=Mock(return_value=torch.zeros(decoded_shape)))
            save_images = Mock(return_value={"ui": {"images": ["thumbnail"]}})
            comfy_nodes = types.ModuleType("nodes")
            comfy_nodes.PreviewImage = Mock(return_value=types.SimpleNamespace(save_images=save_images))
            with patch.dict(sys.modules, {"nodes": comfy_nodes}):
                result = self.nodes._result_with_preview(output, vae)
            torch.testing.assert_close(vae.decode.call_args.args[0], samples[:1])
            self.assertEqual(tuple(save_images.call_args.args[0].shape), (1, 8, 8, 3))
            self.assertIs(result["result"][0], output)
            self.assertEqual(result["ui"]["images"], ["thumbnail"])
            self.assertEqual(output["samples"].shape[0], 3)
        self.assertEqual(self.nodes._result_with_preview(output, None), (output,))

    def test_node_uses_preset_defaults_and_explicit_overrides(self):
        model, positive, negative = object(), object(), object()
        latent = {"samples": torch.zeros(1, 16, 1, 2, 2)}
        run = Mock(return_value=latent)
        with patch.object(self.nodes, "run_sampling", run):
            result = self.nodes.CyberKreaSampler().sample(
                model, positive, latent, 42, "quality", steps=2, sampler="euler_2m",
                contraction=0, negative=negative)
        self.assertIs(result[0], latent)
        args = run.call_args
        self.assertIs(args.args[0], model)
        self.assertIs(args.args[1], positive)
        self.assertIs(args.args[2], negative)
        self.assertIs(args.args[3], latent)
        self.assertEqual(args.args[4].tolist(), [1, .75, 0])
        self.assertEqual(args.kwargs["order"], 2)
        self.assertEqual(args.kwargs["contraction"], 0)
        self.assertEqual(args.kwargs["detail_amount"], .7)


if __name__ == "__main__":
    unittest.main()
