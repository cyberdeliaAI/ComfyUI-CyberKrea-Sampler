# Changelog

## 0.3.0

- Preserve the requested sampling budget for one- and two-step schedules,
  plunge, and disabled restarts. Every generated schedule starts at 1 and ends
  at 0. Restart jumps no longer inflate the preview progress count.
- Honor `sigma_gate` at and above 0.35 using a hard cutoff.
- Correct variable-step AB2 coefficients for manually selected `euler_2m`.
  Quality now displays `euler`, matching the effective behavior of its
  existing stochastic defaults.
- Serve preset and resolution data through Python node metadata instead of
  maintaining duplicate JavaScript tables. Preserve explicit manual values
  and fill omitted values for direct Python callers.
- Add `preview_method=default` for new nodes to follow ComfyUI's global
  setting. Existing explicit preview selections remain valid.
- Decode only the first image for the optional VAE thumbnail, preserving the
  full latent batch output.
- Validate settings and report incompatible latent shapes with an actionable
  error. Honor latent batch indices when preparing initial noise.
- Document the actual CFG ramp and initial-noise contraction semantics.
  Contraction 0 remains available; no unvalidated lower bound is imposed.
- Remove unused upstream features, stale research notes and duplicated README
  text; add upstream attribution while retaining its MIT notice.
- Add CPU tensor tests, frontend behavior checks, and CI for Python 3.11/3.13.

The usual fast, balanced and quality schedules and stochastic sampling math
are preserved. Manual AB2 runs and configurations affected by the corrected
edge cases can produce different results. A real-checkpoint image-quality
validation is not part of the automated tests.
