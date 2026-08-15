# The detection duality: complete measured narrative

Status: synthesis of real-model measurements on the VPS (Qwen2.5-0.5B,
CPU). Produced 2026-08-14. Not release-qualified findings.

## The arc (rounds 31-53)

| Round | Experiment | Result |
| --- | --- | --- |
| 31 | output detection, greedy baseline (P4) | entropy AUC 0.00 (inverted) |
| 31 | output detection, sampled baseline | entropy AUC 0.07, CDF 0.20 (still inverted) |
| 41 | raw surprise feature, corrected direction | stego surprise 0.99 = native expectation; T=0.8 normal 0.45 — **the coder is surprise-preserving; output detection fails by design** |
| 46 | activation probe, group-disjoint, matched baseline | **AUC 1.000** x3 seeds — activations trivial |
| 51 | moderate coupling (0.9/0.9) vs probe | **AUC 0.938** — partial evasion |
| 53 | aggressive coupling (0.7/0.85) vs probe | **AUC 0.333** (at chance; 0 encode failures) — monotone collapse, signal gone |

## The measured access tiers

| Defender | Observable | Measured result |
| --- | --- | --- |
| External observer | output text statistics (surprise, CDF) | **near-chance by design** — the coder preserves the surprise distribution and makes PIT draws uniform |
| Inference provider | hidden states | **trivial uncoupled (1.000)**, partial evasion under moderate coupling (0.938) |

## The three findings

1. **The coder is output-level invisible to statistics, by design, not by
   calibration.** The pseudorandom (encrypted) target makes the coder's
   selections behave like native sampling on the surprise axis
   (mean normalized surprise 0.99 vs the definitional 1.0), and makes PIT
   draws uniform. No amount of recalibration recovers a signal that does
   not exist.

2. **The coder is activation-trivial to an inference provider.** Under the
   leakage-free protocol (group-disjoint prompt split, matched baseline)
   that invalidated the historical claim, the probe still scores 1.000 —
   the historical duality table is restored in corrected form.

3. **Distribution coupling is a monotone, real-model-evidenced evasion.**
   Moderate coupling erodes the probe to 0.938; aggressive coupling
   (0.7/0.85) collapses it to 0.333 at zero encode failures — the evasion
   completes without the predicted capacity cost, and the remaining cost
   is representational: near-deterministic sampling compresses the
   activation manifold until stego and matched-normal are
   indistinguishable. (Small-sample note: with 4 test groups the aggressive
   CI spans 0.0–0.75, so the point is "signal gone", not a measured
   inversion.)

## Implication for defense

The tier that matters is the inference provider: activations detect the
coder trivially unless the attacker pays capacity for coupling, which only
partially helps. External monitoring of output statistics is structurally
impotent against this coder — a defender cannot rely on text-level
detection. The historical "access tiers" framing survives, with both halves
now measured correctly.

Per-experiment detail: `analysis/detection_baseline.md`,
`analysis/recalibrated_detector.md`, `analysis/activation_probe_real.md`,
`analysis/activation_probe_coupled.md`.
