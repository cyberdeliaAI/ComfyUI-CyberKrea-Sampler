# ComfyUI-CyberKrea-Sampler

A compact Krea 2 Turbo sampler and resolution-aware empty latent node for
workflows that already patch the model with LoRAs and/or NegPiP.

CyberKrea uses its own technical identifiers, package, category and display
name, preventing conflicts with other sampler nodes. It provides calibrated
presets with direct manual control while deliberately omitting `clean_model`,
variety, composition blending and the raw/experimental preset.

## Node

`CyberKrea Sampler` is under the `CyberKrea` category.

### Connections

| Option | Description |
|---|---|
| `model` | The final Krea 2 model from your model chain. Connect it after LoRAs and/or NegPiP. |
| `positive` | Positive conditioning from your prompt or NegPiP chain. |
| `latent_image` | A 16-channel Krea 2 / Wan21 latent, such as the output from CyberKrea Empty Latent. |
| `seed` | Controls the initial noise and all seeded sampling-noise streams for reproducible runs. |
| `negative` | Optional negative conditioning. Connecting it enables sigma-window guidance. Leave it disconnected when NegPiP already handles negative concepts. |
| `vae` | Optional. Decodes one thumbnail on the sampler node; it does not change the LATENT output. |

### Sampler controls

Selecting a preset immediately fills in all visible control values. You can
then change any individual value without losing the rest of that preset.

| Option | Description |
|---|---|
| `preset` | Loads `fast`, `balanced`, or `quality` values into every control below. |
| `steps` | Total sampling-step budget. More steps take longer and can add refinement; Krea 2 Turbo usually does not need high counts. |
| `sampler` | `euler` is the direct, stable first-order method. `euler_2m` uses a two-step history (AB2) and is used by the quality preset. |
| `restart_frac` | Fraction of the existing step budget reserved for the restart/detail-recovery pass. Increasing it reallocates steps; it does not add extra steps. |
| `sigma_r` | Noise level to jump back to at the restart. Higher values make the restart stronger; lower values keep it more subtle. |
| `plunge` | Locks the main composition with a direct end step before the lower-sigma restart works on texture and detail. |
| `detail` | Strength of the detail sigma adjustment. Higher values emphasize fine detail; excessive values can look harsh. |
| `eta0` | Maximum gated ancestral-noise strength during the middle of sampling. `0` disables this extra noise injection. |
| `sigma_gate` | Eta noise is disabled below this sigma. Higher values stop ancestral noise earlier; lower values allow it later into the image. |
| `contraction` | Scales the initial noise toward the measured photo manifold. `1.0` is uncontracted stock noise; `0.70` is the preset value. |
| `preview_method` | Live preview: `auto`, fast `latent2rgb`, `taesd`, or `none`. TAESD requires `lighttaew2_1` in `models/vae_approx`. |

Preset defaults:

| Preset | Steps | Detail | Sampler |
|---|---:|---:|---|
| fast | 8 | 0.50 | euler |
| balanced | 12 | 0.60 | euler |
| quality | 16 | 0.70 | euler_2m |

Choosing another preset updates every displayed preset field. You can then
change any individual value without losing the rest of the selected preset.

## CyberKrea Empty Latent

`CyberKrea Empty Latent` creates the 16-channel latent expected by Krea 2 and
outputs the selected `width` and `height` as integers. Choose a size tier first;
the resolution dropdown then shows only the matching dimensions.

| Option | Description |
|---|---|
| `size` | Selects the S, M, L, or XL megapixel tier, filters the resolution dropdown, and preserves the selected aspect ratio. |
| `resolution` | Selects the concrete width, height, and aspect ratio within the chosen tier. |
| `batch_size` | Number of empty latents generated in one batch. Higher values require more VRAM. |

| Tier | Available resolutions |
|---|---|
| S (~1.0 MP) | 1024x1024, 1152x864, 864x1152, 1344x896, 896x1344, 1344x768, 768x1344 |
| M (~1.4 MP) | 1184x1184, 1344x1008, 1008x1344, 1568x1040, 1040x1568, 1568x880, 880x1568 |
| L (~1.7 MP) | 1312x1312, 1504x1120, 1120x1504, 1600x1088, 1088x1600, 1728x960, 960x1728 |
| XL (~2.1 MP) | 1440x1440, 1664x1248, 1248x1664, 1776x1184, 1184x1776, 1920x1088, 1088x1920 |

Every dimension is divisible by 16 for the Wan21 VAE and Krea 2 patch layout.

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

## License

Released under the MIT license. See `LICENSE`.
