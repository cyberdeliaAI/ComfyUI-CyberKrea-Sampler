# ComfyUI-CyberKrea-Sampler

A compact Krea 2 Turbo sampler and resolution-aware empty latent node for
workflows that already patch the model with LoRAs and/or NegPiP.

CyberKrea uses its own technical identifiers, package, category and display
name, preventing conflicts with other sampler nodes. It provides calibrated
presets with direct manual control while deliberately omitting `clean_model`,
variety, composition blending and the raw/experimental preset.

## Node

`CyberKrea Sampler` is under the `CyberKrea` category.

Inputs:

- `model`: connect the final model from your model chain (for example, after
  LoRA and NegPiP).
- `positive`: positive conditioning from your existing prompt/NegPiP chain.
- `latent_image`: your 16-channel Krea 2 / Wan21 latent.
- `seed`
- `preset`: `fast`, `balanced` (default), or `quality`.
- Selecting a preset immediately fills in its real values for `steps`,
  `sampler`, `restart_frac`, `sigma_r`, `plunge`, `detail`, `eta0`,
  `sigma_gate`, and `contraction`. Every populated value can then be adjusted
  directly; there are no sentinel values such as `-1`, `0`, or `preset`.
- `preview_method`: live preview control.
- `negative` (optional): leave disconnected when NegPiP already incorporates
  the negative prompt. Connecting it enables CyberKrea's sigma-window guidance.
- `vae` (optional): shows a decoded thumbnail on the sampler node.

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

| Tier | Available resolutions |
|---|---|
| S (~1.0 MP) | 1024x1024, 1152x864, 896x1344, 1344x768, 768x1344 |
| M (~1.4 MP) | 1184x1184, 1344x1008, 1040x1568, 1568x880, 880x1568 |
| L (~1.7 MP) | 1312x1312, 1504x1120, 1088x1600, 1728x960, 960x1728 |
| XL (~2.1 MP) | 1440x1440, 1664x1248, 1184x1776, 1920x1088, 1088x1920 |

Every dimension is divisible by 16 for the Wan21 VAE and Krea 2 patch layout.

## NegPiP wiring

```text
base model -> LoRA(s) -> NegPiP model output -> CyberKrea Sampler (model)
prompt/NegPiP positive conditioning ----------> CyberKrea Sampler (positive)
CyberKrea Empty Latent -----------------------> CyberKrea Sampler (latent_image)
```

Leave the Lite sampler's `negative` input empty when NegPiP is already handling
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
