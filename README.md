# ComfyUI-CyberKrea-Sampler

A compact Krea 2 Turbo sampler for workflows that already patch the model with
LoRAs and/or NegPiP.

CyberKrea is a separate node and uses its own technical identifiers, package,
category and display name, so it can be installed alongside KreaPhoton without
node conflicts. Its sampling core is derived from
[ComfyUI-KreaPhoton](https://github.com/Kostik2702/ComfyUI-KreaPhoton) and keeps
its Photon sampling engine and calibrated presets. The UI deliberately removes
`clean_model`, variety, composition blending and the raw/experimental preset.

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

## NegPiP wiring

```text
base model -> LoRA(s) -> NegPiP model output -> CyberKrea Sampler (model)
prompt/NegPiP positive conditioning ----------> CyberKrea Sampler (positive)
Krea 2 latent --------------------------------> CyberKrea Sampler (latent_image)
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

## Credits and license

The sampler engine is derived from ComfyUI-KreaPhoton by Kostiantyn Hrytsuk,
used under its MIT license. See `LICENSE`.
