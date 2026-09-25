"""Schedule invariants across the full supported step range."""

import unittest

import torch

from cyberkrea_sampler.presets import PRESETS
from cyberkrea_sampler.schedules import build_schedule, sigma_from_t, t_from_sigma


class ScheduleTests(unittest.TestCase):
    def test_budget_endpoints_and_restart_structure(self):
        for steps in range(1, 65):
            for plunge in (False, True):
                for frac in (0.0, 0.01, 0.25, 0.60):
                    for sigma_r in (0.0, 0.01, 0.65, 1.0):
                        with self.subTest(steps=steps, plunge=plunge, frac=frac, sigma_r=sigma_r):
                            sigmas = build_schedule(steps, restart_frac=frac,
                                                    sigma_r=sigma_r, plunge=plunge)
                            self.assertEqual(sigmas[0].item(), 1.0)
                            self.assertEqual(sigmas[-1].item(), 0.0)
                            self.assertTrue(torch.isfinite(sigmas).all())
                            jumps = (sigmas[1:] > sigmas[:-1]).nonzero().flatten()
                            expected_restart = frac > 0 and sigma_r > 0 and steps > (2 if plunge else 1)
                            self.assertEqual(len(jumps), int(expected_restart))
                            evaluations = ((sigmas[:-1] > 0) & (sigmas[1:] <= sigmas[:-1])).sum()
                            self.assertEqual(evaluations.item(), steps)
                            if expected_restart:
                                self.assertEqual(sigmas[jumps[0]].item(), 0.0)

    def test_one_step_is_always_full_descent(self):
        for plunge in (False, True):
            self.assertEqual(build_schedule(1, restart_frac=.25, sigma_r=.65,
                                            plunge=plunge).tolist(), [1.0, 0.0])

    def test_presets_keep_expected_structure_and_restart_sizes(self):
        for preset in PRESETS.values():
            sigmas = build_schedule(preset["steps"], restart_frac=preset["restart_frac"],
                                    sigma_r=preset["sigma_r"], plunge=preset["plunge"])
            zero_index = (sigmas == 0).nonzero()[0].item()
            self.assertEqual(zero_index, preset["steps"] * 3 // 4)
            self.assertAlmostEqual(sigmas[zero_index - 1].item(), .75)
            self.assertAlmostEqual(sigmas[zero_index + 1].item(), .65)

    def test_shift_inverse(self):
        for alpha in (1.0, 3.158, 10.0):
            for t in (0.0, .1, .5, .9, 1.0):
                self.assertAlmostEqual(t_from_sigma(sigma_from_t(t, alpha), alpha), t)

    def test_invalid_schedule_inputs(self):
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                build_schedule(steps)
        for kwargs in ({"alpha": 0}, {"alpha": float("inf")},
                       {"sigma_r": -1}, {"restart_frac": float("nan")},
                       {"restart_frac": 1.1}):
            with self.assertRaises(ValueError):
                build_schedule(8, **kwargs)


if __name__ == "__main__":
    unittest.main()
