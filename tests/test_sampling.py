"""Tensor-level sampling regressions; no checkpoint or GPU is required."""

import unittest
from unittest.mock import Mock

import torch

from cyberkrea_sampler.noise import contract_noise
from cyberkrea_sampler.presets import PRESETS
from cyberkrea_sampler.schedules import build_schedule
from support import engine_modules


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.modules = self.enterContext(engine_modules())
        self.engine = self.modules.sampling

    def test_loop_obeys_small_step_budgets(self):
        for steps in (1, 2, 3, 4, 8, 12, 16):
            for plunge in (False, True):
                for frac in (0.0, .25, .60):
                    for sigma_r in (0.0, .65):
                        with self.subTest(steps=steps, plunge=plunge, frac=frac, sigma_r=sigma_r):
                            model = Mock(side_effect=lambda x, sigma: x * .5)
                            callback = Mock()
                            sigmas = build_schedule(steps, restart_frac=frac,
                                                    sigma_r=sigma_r, plunge=plunge)
                            result = self.engine.cyberkrea_sampler_loop(
                                model, torch.ones(1, 16, 1, 2, 2), sigmas,
                                callback=callback, eta0=1, order=2)
                            self.assertEqual(model.call_count, steps)
                            self.assertEqual(callback.call_count, steps)
                            self.assertTrue(torch.isfinite(result).all())

    def test_stochastic_streams_are_reproducible(self):
        sigmas = build_schedule(12, restart_frac=.25, sigma_r=.65, plunge=True)
        x = torch.ones(2, 16, 1, 2, 2)
        def run(seed):
            return self.engine.cyberkrea_sampler_loop(
                lambda value, sigma: value.tanh() * .7, x.clone(), sigmas,
                eta0=1, detail_amount=.6, restart_seed=seed)
        first = run(123)
        torch.randn(20)  # Global RNG use must not change the restart stream.
        self.assertTrue(torch.equal(first, run(123)))
        self.assertFalse(torch.equal(first, run(124)))

    def test_default_quality_was_already_effectively_euler(self):
        preset = PRESETS["quality"]
        sigmas = build_schedule(preset["steps"], restart_frac=preset["restart_frac"],
                                sigma_r=preset["sigma_r"], plunge=preset["plunge"])
        results = [self.engine.cyberkrea_sampler_loop(
            lambda x, sigma: x.tanh(), torch.ones(1, 16, 1, 2, 2), sigmas,
            order=order, eta0=preset["eta0"], detail_amount=preset["detail"],
            restart_seed=42) for order in (1, 2)]
        self.assertTrue(torch.equal(*results))

    def test_ab2_accounts_for_unequal_step_widths(self):
        inputs = []
        def model(x, sigma):
            inputs.append(x.clone())
            return x - sigma.square().reshape(-1, 1, 1, 1, 1)
        self.engine.cyberkrea_sampler_loop(
            model, torch.zeros(1, 16, 1, 1, 1), torch.tensor([1., .9, .7, 0.]),
            order=2, eta0=0)
        # d(sigma)=sigma: Euler startup contributes -.1, then integrating
        # the linear derivative from .9 to .7 contributes exactly -.16.
        torch.testing.assert_close(inputs[2], torch.full_like(inputs[2], -.26))

    def test_eta_gate_cuts_off_and_ramps(self):
        eta = self.engine._gated_eta
        for gate in (0, .1, .35, .5, 1):
            self.assertEqual(eta(gate, 1, gate), 0)
            self.assertEqual(eta(gate / 2, 1, gate), 0)
        self.assertAlmostEqual(eta(.225, 1, .1), .5)
        self.assertEqual(eta(.35, 1, .1), 1)
        self.assertEqual(eta(.6, 1, .5), 1)
        self.assertEqual(eta(.2, 1, .5), 0)
        self.assertEqual(eta(.6, 0, .5), 0)

    def test_high_gate_prevents_rng_draws_below_cutoff(self):
        sigmas = torch.tensor([.3, .2, .1, 0.])
        kwargs = dict(model=lambda x, sigma: x * .5,
                      x=torch.ones(1, 16, 1, 2, 2), sigmas=sigmas, sigma_gate=.5)
        first = self.engine.cyberkrea_sampler_loop(**kwargs, eta0=1, restart_seed=42)
        second = self.engine.cyberkrea_sampler_loop(**kwargs, eta0=0, restart_seed=84)
        self.assertTrue(torch.equal(first, second))

    def test_contraction_only_affects_initial_noise(self):
        noise = torch.ones(1, 16, 1, 2, 2)
        self.assertIs(contract_noise(noise, 1), noise)
        self.assertEqual(contract_noise(noise, 0).count_nonzero(), 0)
        torch.testing.assert_close(contract_noise(noise, .7), noise * .7)
        sigmas = build_schedule(4, restart_frac=.25, sigma_r=.65, plunge=True)
        result = self.engine.cyberkrea_sampler_loop(
            lambda x, sigma: x * .5, contract_noise(noise, 0), sigmas, eta0=0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(result.count_nonzero().item(), 0)
        for value in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                contract_noise(noise, value)

    def test_run_keeps_model_conditioning_mask_and_batch_indices(self):
        model = object()
        positive = [[torch.ones(1, 2, 3), {"pooled_output": torch.ones(1, 3)}]]
        negative = [[torch.zeros(1, 2, 3), {}]]
        mask = torch.ones(1, 2, 2)
        latent = {"samples": torch.zeros(2, 16, 1, 2, 2), "noise_mask": mask,
                  "batch_index": [2, 5], "custom": "preserved"}
        sigmas = build_schedule(4, restart_frac=.25, plunge=True)
        for conditioning in (None, negative):
            out = self.engine.run_sampling(model, positive, conditioning, latent, sigmas, seed=123)
            guider = self.modules.guiders[-1]
            self.assertIs(guider.model_patcher, model)
            self.assertIs(guider.conds["positive"], positive)
            self.assertIs(guider.sample.call_args.kwargs["denoise_mask"], mask)
            self.assertEqual(guider.sample.call_args.kwargs["seed"], 123)
            if conditioning is None:
                self.assertEqual(guider.cfg, 1.0)
                self.assertEqual(guider.conds["negative"][0][0].count_nonzero(), 0)
                self.assertEqual(positive[0][0].count_nonzero(), 6)
            else:
                self.assertIsInstance(guider, self.modules.guidance.CyberKreaGuider)
                self.assertIs(guider.conds["negative"], negative)
            self.assertEqual(self.modules.comfy.sample.prepare_noise.call_args.args[2], [2, 5])
            self.assertIsNot(out, latent)
            self.assertIs(out["noise_mask"], mask)
            self.assertEqual(out["custom"], "preserved")
            self.assertEqual(latent["samples"].count_nonzero(), 0)

    def test_preview_progress_excludes_restart_jump(self):
        model = object()
        self.engine.run_sampling(model, [], None, {"samples": torch.zeros(1, 16, 1, 2, 2)},
                                 build_schedule(4, restart_frac=.25, plunge=True), seed=0)
        self.modules.preview.prepare_callback.assert_called_once_with(model, 4)
        callback = self.modules.guiders[-1].sample.call_args.kwargs["callback"]
        for index in (0, 1, 2, 4):
            callback(index, "denoised", "latent", 5)
        self.assertEqual([call.args for call in self.modules.preview.prepare_callback.return_value.call_args_list],
                         [(index, "denoised", "latent", 4) for index in range(4)])

    def test_wrong_latent_shape_has_actionable_error(self):
        for shape in ((1, 4, 2, 2), (1, 4, 1, 2, 2), (1, 16, 2, 2, 2)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "Krea 2 / Wan21"):
                self.engine.run_sampling(object(), [], None, {"samples": torch.zeros(shape)},
                                         build_schedule(1), seed=0)

    def test_comfy_converts_empty_image_latent_before_shape_validation(self):
        latent = {"samples": torch.zeros(1, 16, 2, 2)}
        self.modules.comfy.sample.fix_empty_latent_channels.side_effect = (
            lambda model, samples: samples.unsqueeze(2))
        result = self.engine.run_sampling(object(), [], None, latent, build_schedule(1), seed=0)
        self.assertEqual(tuple(result["samples"].shape), (1, 16, 1, 2, 2))
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 2, 2))


if __name__ == "__main__":
    unittest.main()
