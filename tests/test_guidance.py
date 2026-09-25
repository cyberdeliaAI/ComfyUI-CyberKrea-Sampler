"""Guidance shape and forwarding through the patched model path."""

import unittest

import torch

from support import engine_modules


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())

    def test_cfg_ramp_and_plateau(self):
        g = self.modules.guidance.g_window
        for sigma in (0.0, .65, .7):
            self.assertEqual(g(sigma, 1.25), 1.0)
        self.assertAlmostEqual(g(.8, 1.25), 1.625)
        for sigma in (.9, 1.0, 1.1):
            self.assertEqual(g(sigma, 1.25), 2.25)
        with self.assertRaises(ValueError):
            g(.8, 1.25, lo=.9, hi=.7)

    def test_guider_forwards_model_patches_seed_and_conditioning(self):
        model = object()
        guider = self.modules.guidance.CyberKreaGuider(model, 1.25)
        positive, negative = object(), object()
        guider.set_conds(positive, negative)
        options = {"transformer_options": {"patches": object()}}
        x = torch.ones(1)
        for sigma, expected in ((.5, 1), (1.0, 2.25)):
            timestep = torch.tensor([sigma])
            guider.predict_noise(x, timestep, model_options=options, seed=42)
            args = self.modules.comfy.samplers.sampling_function.call_args
            self.assertIs(args.args[0], model)
            self.assertIs(args.args[3], negative)
            self.assertIs(args.args[4], positive)
            self.assertEqual(args.args[5], expected)
            self.assertIs(args.kwargs["model_options"], options)
            self.assertEqual(args.kwargs["seed"], 42)


if __name__ == "__main__":
    unittest.main()
