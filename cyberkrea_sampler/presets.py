"""
Single source of truth for every calibrated constant (planning-council R19).

No other module hardcodes a preset value or a manifold constant. Values marked
"CALIBRATE: V<n>" are starting hypotheses from docs/04 design math (M1-M6) and
research/results/E*.json, pending the corresponding validation slot in
docs/05-plan-kreaphoton-v1.md §V-protocol (S8.5/S10) — nothing here is final
until that V-slot freezes it. Values NOT marked CALIBRATE are measured facts
(e.g. MANIFOLD_STD/MEAN from research/results/E3_real.json), not guesses.
"""

# --- Manifold constants (MEASURED, research/results/E3_real.json normalized_per_channel,
#     8 real photographs through qwen_image_vae; global std=0.4666, mean=-0.0070) ---
MANIFOLD_STD = [
    0.34837067127227783, 0.34285303950309753, 0.45075058937072754, 0.4954315721988678,
    0.41811972856521606, 0.4532100260257721, 0.3540203869342804, 0.477848619222641,
    0.3242858350276947, 0.3410053551197052, 0.31148362159729004, 0.38901352882385254,
    0.3583352863788605, 0.4181098937988281, 0.36241018772125244, 0.4790935218334198,
]
MANIFOLD_MEAN = [
    0.28196287155151367, 0.40890082716941833, -0.0024451527278870344, -0.08094137161970139,
    -0.20453284680843353, -0.2656249701976776, -0.09275273233652115, -0.4799724817276001,
    -0.24426628649234772, -0.19706667959690094, -0.037885453552007675, 0.37430068850517273,
    0.10810646414756775, -0.04556984826922417, 0.2223597764968872, 0.14363540709018707,
]

# --- Guidance window (M6). V1/E1b RAN 2026-07-05 (docs/06 pre-registered protocol,
#     27 img: {1.25,1.31,1.5} x P1/P2/P3 x seed1001-1003): delta=1.5 catastrophically
#     hallucinates (P1: paint-like mutation on shoulder; P2: ghost face/eyes overlaid
#     on the whole scene - worse than the E1 "crocodile skin" early-warning sign, a
#     full compositional break, not a mild degrade). delta=1.25 and 1.31 both clean
#     across all 9 cells each; per pre-registered tie-break (smallest passing
#     candidate) delta=1.25 wins. Gate satisfied -> window enabled by default now. ---
GUIDANCE = {
    "delta": 1.25,              # V1-VALIDATED (was CALIBRATE placeholder, same value)
    "lo": 0.7,
    "hi": 0.9,
    "enabled_by_default": True,  # V1 gate passed (docs/04: "включается по умолчанию только после E1b")
    "flat_cfg": 1.15,           # fallback for guidance_mode="flat" (explicit full-traj cfg,
                                 # e.g. Advanced node override); "window" is now the Sampler default.
}

# Photo resolution buckets (Wan21 spatial /8 x DiT patch 2x2 -> effective /16;
# every dim below is divisible by 16). Canon anchor: user's ErikaNew4 baseline
# 1088x1600 (2:3 portrait, docs/01 §5) sits in the L tier.
RESOLUTION_BUCKETS = {
    "S (~1.0 MP)": {
        "1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152),
        "3:2": (1344, 896), "2:3": (896, 1344),
        "16:9": (1344, 768), "9:16": (768, 1344),
    },
    "M (~1.4 MP)": {
        "1:1": (1184, 1184), "4:3": (1344, 1008), "3:4": (1008, 1344),
        "3:2": (1568, 1040), "2:3": (1040, 1568),
        "16:9": (1568, 880), "9:16": (880, 1568),
    },
    "L (~1.7 MP)": {
        "1:1": (1312, 1312), "4:3": (1504, 1120), "3:4": (1120, 1504),
        "3:2": (1600, 1088), "2:3": (1088, 1600),
        "16:9": (1728, 960), "9:16": (960, 1728),
    },
    "XL (~2.1 MP)": {
        "1:1": (1440, 1440), "4:3": (1664, 1248), "3:4": (1248, 1664),
        "3:2": (1776, 1184), "2:3": (1184, 1776),
        "16:9": (1920, 1088), "9:16": (1088, 1920),
    },
}
RESOLUTION_ASPECTS = list(RESOLUTION_BUCKETS["L (~1.7 MP)"].keys())
DEFAULT_RESOLUTION_SIZE = "L (~1.7 MP)"
DEFAULT_RESOLUTION_ASPECT = "2:3"

# --- Presets ---
PRESETS = {
    "turbo/fast": {
        "n_steps": 8,
        "alpha": 3.158,             # e^1.15, stock ModelSamplingFlux shift; critic-verified base
        "restart_frac": 0.25,       # V2-VALIDATED (kept at design hypothesis, 54-img grid didn't test frac itself)
        "sigma_r": 0.65,            # V2-VALIDATED: {0.55,0.60,0.65}x{plunge} grid, 2026-07-05 -
                                     # higher sigma_r recovers shadow/fabric texture (dark_blotch
                                     # +25..98% vs no-restart baseline = RECOVERED DETAIL, confirmed
                                     # visually clean on P2 zoom-crops, not blotch/noise artifacts;
                                     # P1 face crop: freckles naturally varied, no stamping)
        "plunge": True,             # V2-VALIDATED: plunge=True consistently reduces/reverses the
                                     # lap_sharpness regression vs baseline at every sigma_r tested
        "detail_a": 0.50,           # UNTESTED (M2 working range 0.4-0.8; held constant across
                                     # all of V1-V6 - no dedicated V-slot tested this value; not
                                     # "V2" as previously mislabeled - V2 only swept sigma_r/plunge)
        "eta0": 1.0,                # V4-VALIDATED: {0.0,0.5,1.0} x P2 x seed1001-1003 (quality
                                     # preset, N=16), 2026-07-05 - eta0=1.0 gave the CLEANEST
                                     # shadows (lowest hf_noise/dark_blotch of the 3, not just
                                     # visually artifact-free) - highest tested value wins per
                                     # accept-rule, no regression found at this sr=0.65/plunge=True
                                     # config (was the open M5 event-hypothesis risk; resolved safe)
        "sigma_gate": 0.10,         # M5-proven terminal cutoff
        "contraction": 0.70,        # V5-VALIDATED, see turbo/balanced comment
        "sampler": "euler",
    },
    "turbo/balanced": {             # DEFAULT
        "n_steps": 12,
        "alpha": 3.158,
        "restart_frac": 0.25,
        "sigma_r": 0.65,            # V2-VALIDATED, see turbo/fast comment
        "plunge": True,             # V2-VALIDATED
        "detail_a": 0.60,
        "eta0": 1.0,
        "sigma_gate": 0.10,
        "contraction": 0.70,        # V5-VALIDATED: {1.00,0.85,0.70} x P1/P2/P3 x seed1001-1003,
                                     # 2026-07-05 - accept-rule is "most aggressive value with no
                                     # inter-seed SSIM regression vs c=1.00 control" (guards against
                                     # the E2 alpha-tightening "sameness" mechanism). Neither 0.85
                                     # (SSIM +0.9%, flat) nor 0.70 (SSIM -4.7%, MORE diverse, not
                                     # less) regressed - both pass, picked smallest per tie-break.
                                     # hf_noise/dark_blotch improve monotonically with stronger
                                     # contraction (cleaner, not "flatter"); P1/P3 visual check
                                     # confirms no loss of photographic naturalness at 0.70.
        "sampler": "euler",
    },
    "turbo/quality": {
        "n_steps": 16,
        "alpha": 3.158,
        "restart_frac": 0.25,
        "sigma_r": 0.65,            # V2-VALIDATED
        "plunge": True,             # V2-VALIDATED
        "detail_a": 0.70,
        "eta0": 1.0,                # mandatory gated-eta at this step count (design: "gated-eta обязателен")
        "sigma_gate": 0.10,
        "contraction": 0.70,        # V5-VALIDATED, see turbo/balanced comment
        "sampler": "euler_2m",
    },
    "raw/experimental": {
        "n_steps": 36,
        "alpha": 3.158,             # CALIBRATE: raw canon is dynamic mu 0.5->1.15 (docs/01); unexplored
        "restart_frac": 0.20,
        "sigma_r": 0.45,
        "plunge": False,
        "detail_a": 0.50,
        "eta0": 1.0,
        "sigma_gate": 0.10,
        "contraction": 1.00,       # no contraction calibration attempted for RAW yet
        "sampler": "euler_2m",
        "cfg": 3.5,                # RAW needs real CFG (negative works, unlike Turbo)
        "experimental": True,
    },
}

DEFAULT_PRESET = "turbo/balanced"
