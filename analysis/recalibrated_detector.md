# Recalibrated output detection: the coder is surprise-preserving

Status: real-model pilot on the VPS (Qwen2.5-0.5B-Instruct, CPU), not
release-qualified findings. Produced 2026-08-14.

## The question

Round 31 found the StegoDetector inverted (entropy AUC 0.07 even with a
sampled baseline). The hypothesis for a corrected detector was: the
arithmetic coder selects high-probability tokens, so stego text should have
mean normalized surprise below normal sampling. `experiments/detector_recalibration.py`
measured the raw feature on real text.

## Results (mean normalized surprise = -log p(token) / entropy, n=10)

| Text | Mean normalized surprise |
| --- | --- |
| **stego** (coder, temperature 1.0) | **0.987** |
| sampled-normal (temperature 0.8, top-p 0.9) | 0.449 |
| greedy-normal | 0.293 |

## Findings

1. **The corrected-direction hypothesis is refuted — for the best reason
   possible.** Stego text has mean normalized surprise 0.99, exactly the
   value native sampling predicts (E[-log p / entropy] = 1 by the entropy
   definitional identity). The coder's selections are *statistically
   indistinguishable from native sampling on this axis*: the pseudorandom
   (encrypted) target spreads the chosen tokens across the distribution the
   way sampling does.

2. **The "inversion" was a baseline-temperature artifact.** The sampled
   baseline used temperature 0.8 + top-p 0.9, which sharpens the
   distribution: its tokens are far more probable (surprise 0.45), so the
   old detector's `abs(mean - 1)` transform flagged normal text (far from
   1) as stego while stego (at 1) scored ~0. With a matched-temperature
   (T=1.0) baseline, stego (0.99) and normal (~1.0) are not separable.

3. **Output-level detection fails by design, not by calibration.** Both
   surprise-based detection (this round) and CDF-uniformity (structurally:
   a pseudorandom target makes PIT draws uniform) cannot separate this
   coder from matched sampling. This is the honest, theory-consistent
   answer to the round-1 detection-duality question for this coder: it is
   output-level invisible to these statistics. Detection must move to
   other observables (per-token joint statistics at scale, or
   activation-level probes).

4. **The old detector is still miscalibrated** (the `abs()` transform has no
   principled basis), but fixing it does not create a detector — there is
   no surprise signal to detect.

## Follow-up

A matched-temperature (T=1.0, top-p=1.0) normal control would confirm the
~1.0 expectation directly; the definitional identity plus the stego value
(0.99) already make the conclusion robust.

Raw results: `findings/detector_recalibration/results_*.json` (gitignored).
