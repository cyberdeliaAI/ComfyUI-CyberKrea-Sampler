# Contributing

Bug reports and focused pull requests are welcome.

Before opening a pull request:

1. Keep existing node IDs and Registry IDs stable unless a breaking change is intentional.
2. Add or update tests for behavior changes.
3. With Python 3.11+ and Node.js 22+ installed, install PyTorch and NumPy in
   a test environment and run `python -B -m unittest discover -s tests -v`.
4. Do not include models, generated checkpoints, credentials, or private workflow data.

Please describe the problem, the expected behavior, and the ComfyUI version used.

The tests execute the schedule and sampler with real CPU tensors and small
model functions. ComfyUI integration boundaries use test doubles. Frontend
checks run the actual JavaScript extensions with Python's node metadata.
These checks do not replace a Krea 2 generation in ComfyUI when assessing
image quality or compatibility with third-party model patches.
