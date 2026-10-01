# ComfyUI-CyberKrea-Sampler

<p align="center">
  <img src="assets/icon.png" alt="CyberKrea Sampler icon" width="200">
</p>

[![Tests](https://github.com/cyberdeliaAI/ComfyUI-CyberKrea-Sampler/actions/workflows/tests.yml/badge.svg)](https://github.com/cyberdeliaAI/ComfyUI-CyberKrea-Sampler/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A compact Krea 2 Turbo sampler and resolution-aware empty latent node for
workflows that already patch the model with LoRAs and/or NegPiP.

One model connection keeps the model supplied by your LoRA/NegPiP chain active
throughout sampling. Choose a preset, then adjust any control independently.
CyberKrea uses its own node identifiers and category so it can coexist with
other sampler packs.

The sampling engine is derived from
[ComfyUI-KreaPhoton](https://github.com/Kostik2702/ComfyUI-KreaPhoton)
by Kostiantyn Hrytsuk. CyberKrea adds a compact interface, a single model path,
resolution selection and its own maintenance fixes.

## Node

`CyberKrea Sampler` is under the `CyberKrea` category.

### Connections

| Option | Description |
|---|---|
| `model` | The final Krea 2 model from your model chain. Connect it after LoRAs and/or NegPiP. |
| `positive` | Positive conditioning from your prompt or NegPiP chain. |
| `latent_image` | A 16-channel Krea 2 / Wan21 latent, such as the output from CyberKrea Empty Latent. |
| `seed` | Controls the initial noise and all seeded sampling-noise streams for reproducible runs. |
| `negative` | Optional negative conditioning. Connecting it enables the sigma-dependent CFG ramp described below. Leave it disconnected when NegPiP already handles negative concepts. |
| `vae` | Optional. Fully decodes the first image in the batch and displays a thumbnail on the node. This costs a full VAE decode for that image; the complete LATENT output is unchanged. |

### Sampler controls

Selecting a preset immediately fills in all visible control values. You can
then change any individual value without losing the rest of that preset.

| Option | Description |
|---|---|
| `preset` | Loads `fast`, `balanced`, or `quality` sampling values into the controls. The preview choice is preserved. |
| `steps` | Number of denoising evaluations in the sampler loop, from 1 to 64. CFG and model patches can require additional work within each evaluation. |
| `sampler` | `euler` is the first-order method used by all presets. `euler_2m` uses variable-step Adams–Bashforth history on consecutive deterministic steps; set `eta0=0` to explore this mode. Restarts and stochastic steps reset its history. |
| `restart_frac` | Fraction of the existing step budget reserved for the restart pass. The restart is skipped if the budget cannot fit both descents. |
| `sigma_r` | Noise level to jump back to at the restart. Higher values make the restart stronger. `0` disables restart and keeps the entire step budget for the first descent. |
| `plunge` | Locks the main composition with a direct end step before the lower-sigma restart works on texture and detail. |
| `detail` | Strength of the detail sigma adjustment. Higher values emphasize fine detail; excessive values can look harsh. |
| `eta0` | Maximum gated ancestral-noise strength during the middle of sampling. `0` disables this extra noise injection. |
| `sigma_gate` | Eta noise is disabled at or below this sigma. Below `0.35`, strength ramps up to `eta0` at sigma `0.35`. A gate of `0.35` or higher uses a hard cutoff. |
| `contraction` | Multiplies the initial noise: `1.0` leaves it unchanged, `0.70` is the preset value, and `0` removes it. Restart and ancestral noise are separate. Very low values are experimental. |
| `preview_method` | `default` follows ComfyUI's global preview setting. Explicit overrides are `auto`, `latent2rgb`, `taesd`, and `none`. TAESD requires `lighttaew2_1` in `models/vae_approx`. The previous global setting is restored after sampling. |

Preset defaults:

| Preset | Steps | Detail | Sampler |
|---|---:|---:|---|
| fast | 8 | 0.50 | euler |
| balanced | 12 | 0.60 | euler |
| quality | 16 | 0.70 | euler |

All presets use `restart_frac=0.25`, `sigma_r=0.65`, `plunge=true`, `eta0=1.0`,
`sigma_gate=0.10`, and `contraction=0.70`. Quality previously displayed
`euler_2m`, but these stochastic defaults never used its second-order step;
the updated sampler selection describes the same effective sampling behavior.

One step always uses the full `[1, 0]` descent. A restart needs at least two
steps without plunge, or three with plunge. Smaller budgets fall back to a
single descent while preserving the requested evaluation count.

With a `negative` input connected, CFG is exactly `1.0` at sigma ≤ `0.7`,
smoothly rises to `2.25` between `0.7` and `0.9`, and stays `2.25` above `0.9`.
This is a ramp with a plateau, not a bounded window. ComfyUI's CFG=1 optimization
can apply at low sigma unless a model patch disables it. Without `negative`,
the sampler uses CFG `1.0` throughout.

### API and saved workflows

The frontend obtains preset and resolution tables from Python's node metadata;
there are no separate JavaScript copies. Existing workflows keep their saved
control values and explicit preview choices. To follow the global preview
setting in an older workflow, choose `default` yourself.

ComfyUI API prompts must include the visible required controls. The supplied
values take precedence: changing only the `preset` label does not overwrite
them. Headless clients can obtain the templates from
`GET /object_info/CyberKreaSampler`, at
`CyberKreaSampler.input.required.preset[1].cyberkrea_presets`, and merge their
overrides before submitting the prompt. Direct Python callers can use
`resolve_settings("quality", steps=20)` to fill omitted values from a preset.

## CyberKrea Empty Latent

`CyberKrea Empty Latent` creates the 16-channel latent expected by Krea 2 and
outputs the selected `width` and `height` as integers. Choose a size tier first;
the resolution dropdown then shows only the matching dimensions.

| Option | Description |
|---|---|
| `size` | Selects the S, M, L, or XL megapixel tier, filters the resolution dropdown, and preserves the selected aspect ratio. |
| `resolution` | Selects the concrete width, height, and aspect ratio within the chosen tier. |
| `batch_size` | Number of empty latents generated in one batch. Higher values require more VRAM. |

| Aspect | S (~1.0 MP) | M (~1.4 MP) | L (~1.7 MP) | XL (~2.1 MP) |
|---|---|---|---|---|
| 1:1 | 1024x1024 | 1184x1184 | 1312x1312 | 1440x1440 |
| 4:3 | 1152x864 | 1344x1008 | 1504x1120 | 1664x1248 |
| 3:4 | 864x1152 | 1008x1344 | 1120x1504 | 1248x1664 |
| 3:2 | 1344x896 | 1568x1040 | 1600x1088 | 1776x1184 |
| 2:3 | 896x1344 | 1040x1568 | 1088x1600 | 1184x1776 |
| 16:9 | 1344x768 | 1568x880 | 1728x960 | 1920x1088 |
| 9:16 | 768x1344 | 880x1568 | 960x1728 | 1088x1920 |
| 21:9 | 1568x672 | 1792x768 | 2016x864 | 2240x960 |
| 9:21 | 672x1568 | 768x1792 | 864x2016 | 960x2240 |
| 5:4 | 1120x896 | 1360x1088 | 1440x1152 | 1600x1280 |
| 4:5 | 896x1120 | 1088x1360 | 1152x1440 | 1280x1600 |

Every dimension is divisible by 16 for the Wan21 VAE and Krea 2 patch layout.
Some original buckets approximate their displayed aspect ratio; the added
21:9, 9:21, 5:4 and 4:5 buckets are exact. These additions extend the available
canvas sizes; their image quality has not been validated through generation tests.

## NegPiP wiring

```text
base model -> LoRA(s) -> NegPiP model output -> CyberKrea Sampler (model)
prompt/NegPiP positive conditioning ----------> CyberKrea Sampler (positive)
CyberKrea Empty Latent -----------------------> CyberKrea Sampler (latent_image)
```

Leave the sampler's `negative` input empty when NegPiP is already handling
negative concepts.

## Installation

Copy the complete `ComfyUI-CyberKrea-Sampler` folder to:

```text
ComfyUI/custom_nodes/ComfyUI-CyberKrea-Sampler
```

Restart ComfyUI. No extra Python packages are required beyond ComfyUI's own
PyTorch/Comfy modules.

After updating, also refresh the browser so it loads the matching frontend
extensions and node metadata. See [CHANGELOG.md](CHANGELOG.md) for changes.

## Development checks

With Python 3.11+, PyTorch, NumPy and Node.js 22+ installed:

```sh
python -B -m unittest discover -s tests -v
```

Tests cover schedule budgets, tensor-level sampling, seed reproducibility,
guidance, previews, validation and frontend widget behavior. ComfyUI boundaries
use test doubles; image quality and third-party patch compatibility still need
generation tests with a real Krea 2 checkpoint.

## License

Released under the MIT license. The original Kostiantyn Hrytsuk copyright
notice is retained alongside Cyberdelia's notice. See [LICENSE](LICENSE).
