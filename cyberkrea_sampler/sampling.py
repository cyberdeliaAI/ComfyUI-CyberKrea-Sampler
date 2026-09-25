"""Euler/AB2 sampling with detail adjustment, restart and gated ancestral noise.

One guider keeps the workflow's patched model active for the full trajectory.
The engine is derived from ComfyUI-KreaPhoton; see README for attribution.
"""
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import torch

from .guidance import CyberKreaGuider
from .noise import contract_noise


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _detail_envelope(p: float, start: float, end: float, peak: float) -> float:
    if end <= start:
        return 0.0
    u = (p - start) / (end - start)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    peak = max(0.05, min(0.95, peak))
    w = u / peak if u < peak else (1.0 - u) / (1.0 - peak)
    return _smoothstep(w)


GATE_HI = 0.35


def _gated_eta(sigma_next: float, eta0: float, sigma_gate: float, gate_hi: float = GATE_HI) -> float:
    """No noise at/below the gate; ramp to eta0 at gate_hi.

    A gate at/above gate_hi uses a hard cutoff because the ramp has no width.
    """
    if sigma_next <= sigma_gate:
        return 0.0
    if gate_hi <= sigma_gate:
        return eta0
    u = (sigma_next - sigma_gate) / (gate_hi - sigma_gate)
    return eta0 * _smoothstep(u)


@torch.no_grad()
def cyberkrea_sampler_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                            detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                            order=1, eta0=0.0, sigma_gate=0.10, restart_seed=0):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    gen = torch.Generator(device="cpu").manual_seed((int(restart_seed) + 0x5EED) & 0xffffffffffffffff)
    old_d = None
    old_dt = None

    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])

        # Schedules encode a restart as an ascending jump from zero.
        if s_next > s_cur:
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (1.0 - s_next) * x + s_next * eps
            old_d = None
            old_dt = None
            continue

        if s_cur <= 0.0:
            continue

        # Skip the detail adjustment on large jumps and the final readout.
        p = i / max(n - 1, 1)
        is_final = s_next <= 0.0
        is_plunge = (s_cur - s_next) > 0.25
        if is_final or is_plunge:
            a = 0.0
        else:
            a = detail_amount * _detail_envelope(p, detail_start, detail_end, detail_peak)
            a = max(-1.0, min(1.0, a))
        sigma_model = max(1e-4, s_cur - a * (s_cur - s_next))

        denoised = model(x, sigma_model * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})

        if is_final:
            x = denoised
            old_d = None
            old_dt = None
            continue

        eta = _gated_eta(s_next, eta0, sigma_gate)
        if eta > 0.0:
            # ComfyUI's rectified-flow ancestral update. A stochastic jump
            # invalidates the deterministic derivative history used by AB2.
            downstep_ratio = 1.0 + (s_next / s_cur - 1.0) * eta
            sigma_down = s_next * downstep_ratio
            alpha_next = 1.0 - s_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = max(0.0, s_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2) ** 0.5
            ratio = sigma_down / s_cur
            x = ratio * x + (1.0 - ratio) * denoised
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (alpha_next / alpha_down) * x + eps * renoise_coeff
            old_d = None
            old_dt = None
        else:
            d = (x - denoised) / s_cur
            dt = s_next - s_cur
            if order >= 2 and old_d is not None and abs(dt) <= 0.25:
                # Variable-step AB2; 1.5*d - 0.5*old_d only holds for equal dt.
                d_use = d + (dt / (2.0 * old_dt)) * (d - old_d)
            else:
                d_use = d
            x = x + d_use * dt
            old_d = d
            old_dt = dt

    return x


def zero_conditioning(cond):
    """Copy positive conditioning with zeroed embeddings for CFG=1 sampling."""
    out = []
    for t, d in cond:
        d = d.copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t), d])
    return out


def run_sampling(model, positive, negative, latent_dict, sigmas, *, seed,
                 delta=1.25, lo=0.7, hi=0.9, contraction=1.0,
                 detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                 order=1, eta0=0.0, sigma_gate=0.10):
    """Sample one Krea 2 image batch using the same patched model throughout."""
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    if latent.ndim != 5 or latent.shape[1] != 16 or latent.shape[2] != 1:
        raise ValueError(
            "CyberKrea requires a Krea 2 / Wan21 image latent with shape "
            f"(batch, 16, 1, height, width); got {tuple(latent.shape)}. "
            "Connect a Krea 2 model and a compatible latent, such as CyberKrea Empty Latent."
        )

    use_guidance = negative is not None
    if negative is None:
        negative = zero_conditioning(positive)

    noise = comfy.sample.prepare_noise(latent, seed, latent_dict.get("batch_index"))
    noise = contract_noise(noise, strength=contraction)

    sampler = comfy.samplers.KSAMPLER(cyberkrea_sampler_loop, extra_options={
        "detail_amount": float(detail_amount), "detail_start": float(detail_start),
        "detail_end": float(detail_end), "detail_peak": float(detail_peak),
        "order": int(order), "eta0": float(eta0), "sigma_gate": float(sigma_gate),
        "restart_seed": int(seed),
    })

    if use_guidance:
        guider = CyberKreaGuider(model, delta=delta, lo=lo, hi=hi)
    else:
        guider = comfy.samplers.CFGGuider(model)
        guider.set_cfg(1.0)
    guider.set_conds(positive, negative)

    try:
        import latent_preview
        sampling_steps = int(((sigmas[:-1] > 0) & (sigmas[1:] <= sigmas[:-1])).sum())
        preview_callback = latent_preview.prepare_callback(model, sampling_steps)
        completed = 0

        def callback(_step, denoised, x, _total):
            nonlocal completed
            preview_callback(completed, denoised, x, sampling_steps)
            completed += 1
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(noise, latent, sampler, sigmas,
                            denoise_mask=latent_dict.get("noise_mask"),
                            callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(),
                         dtype=comfy.model_management.intermediate_dtype())
    out = latent_dict.copy()
    out["samples"] = samples
    return out
