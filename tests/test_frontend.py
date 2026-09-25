"""Run frontend behavior checks using metadata from the actual Python nodes."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from cyberkrea_sampler.resolutions import CyberKreaEmptyLatent
from support import engine_modules


class FrontendTests(unittest.TestCase):
    def test_widgets_use_backend_metadata_and_preserve_overrides(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node.js is required for the frontend tests")
        with engine_modules() as modules:
            metadata = {
                "sampler": {"name": "CyberKreaSampler", "input": modules.nodes.CyberKreaSampler.INPUT_TYPES()},
                "resolutions": {"name": "CyberKreaEmptyLatent", "input": CyberKreaEmptyLatent.INPUT_TYPES()},
            }
        result = subprocess.run([node, str(Path(__file__).with_name("test_frontend.mjs"))],
                                input=json.dumps(metadata), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
