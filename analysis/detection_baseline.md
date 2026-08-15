# Output-level detection with a proper sampling baseline

Status: real-model pilot on the VPS (Qwen2.5-0.5B-Instruct, CPU), not
release-qualified findings. Produced 2026-08-14.

## The question

The overnight P4 phase measured entropy/CDF detection AUC using *greedy*
normal text and found the detector inverted (AUC 0.0). The round-1
detection-duality hypothesis assumed a normal *sampling* baseline. This
experiment (`experiments/detection_baseline_eval.py`, n=10 prompts) adds a
proper temperature-sampled baseline (0.8, top-p 0.9) alongside greedy.

## Results (StegoDetector AUC)

| Detector | stego vs sampled-normal | stego vs greedy-normal |
| --- | --- | --- |
| entropy | **0.07** | 0.00 |
| CDF | **0.20** | 0.17 |

## Findings

1. **The inversion is not a baseline artifact — it is the detector.**
   With the proper sampled baseline the entropy detector is still
   near-perfectly inverted (AUC 0.07), and CDF is weakly inverted (0.20).
   The StegoDetector's score mapping (`abs(mean(normalized_surprise) - 1)`)
   systematically flags *normal sampled* text as stego and *stego* text as
   normal, on real 0.5B data, under both baselines.

2. **The round-1 detection-duality hypothesis, as implemented in
   `detector.py`, is refuted on this data** — and the follow-up
   (`analysis/recalibrated_detector.md`) shows why it is refuted *by
   design*: the coder's mean normalized surprise is 0.99 (native-sampling
   expectation), while the T=0.8 baseline sat at 0.45, so the `abs(mean-1)`
   transform flagged the *sharpened normal* text as stego. Surprise-based
   detection cannot separate this coder from matched-temperature sampling.

3. **The negative is robust and consistent.** stego-vs-greedy entropy AUC
   0.000 reproduces P4 exactly; the corrected baseline moves the number
   from 0.000 to 0.070 — directionally the same. The finding holds at
   n=10 on a small model; a capable model (larger, sampled) is the
   follow-up, but the detector's mapping flaw is structural, not
   model-scale.

Raw results: `findings/detection_baseline/results_*.json` (gitignored).
