"""Resolution tiers, stale widget recovery and real latent allocation."""

import unittest

from cyberkrea_sampler import resolutions


class ResolutionTests(unittest.TestCase):
    def test_tiers_dimensions_and_metadata(self):
        self.assertEqual(list(resolutions.RESOLUTION_OPTIONS),
                         ["S (~1.0 MP)", "M (~1.4 MP)", "L (~1.7 MP)", "XL (~2.1 MP)"])
        self.assertEqual(resolutions.DEFAULT_RESOLUTION, "1088x1600 (2:3)")
        self.assertEqual(resolutions.CyberKreaEmptyLatent.INPUT_TYPES()["required"]["size"][1]
                         ["cyberkrea_resolutions"], resolutions.RESOLUTION_OPTIONS)
        for tier, options in resolutions.RESOLUTION_OPTIONS.items():
            self.assertEqual(len(options), 11)
            for option in options:
                width, height = resolutions.resolve_dimensions(tier, option)
                self.assertEqual(width % 16, 0)
                self.assertEqual(height % 16, 0)
                self.assertTrue(option.startswith(f"{width}x{height} "))

    def test_stale_resolution_preserves_each_aspect_in_each_tier(self):
        for source in resolutions.ALL_RESOLUTIONS:
            aspect = source.rpartition("(")[2].removesuffix(")")
            for tier in resolutions.RESOLUTION_OPTIONS:
                self.assertEqual(resolutions.resolve_dimensions(tier, source),
                                 resolutions.RESOLUTION_BUCKETS[tier][aspect])

    def test_unknown_size_or_resolution_is_rejected(self):
        for tier, label in [("unknown", resolutions.DEFAULT_RESOLUTION),
                            ("XL (~2.1 MP)", "invalid (3:2)")]:
            with self.assertRaises(ValueError):
                resolutions.resolve_dimensions(tier, label)

    def test_generate_allocates_zero_latents(self):
        latent, width, height = resolutions.CyberKreaEmptyLatent().generate(
            "S (~1.0 MP)", "768x1344 (9:16)", 2)
        self.assertEqual((width, height), (768, 1344))
        self.assertEqual(tuple(latent["samples"].shape), (2, 16, 168, 96))
        self.assertEqual(latent["samples"].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
