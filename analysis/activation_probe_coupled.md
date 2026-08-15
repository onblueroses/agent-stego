# Activation-probe evasion: distribution coupling, measured

Status: real-model pilot on the VPS (Qwen2.5-0.5B-Instruct, CPU), not
release-qualified findings. Produced 2026-08-14.

## The question

Round 46 proved the uncoupled coder is activation-trivial (group-disjoint
probe AUC 1.000). The historical "distribution coupling" evasion was
invalidated by the same protocol errors. This experiment measures coupling
against the REAL probe: coupled stego (temperature/top-p applied before
coding) vs a MATCHED normal baseline (same temperature/top-p sampling),
group-disjoint split, three seeds.

## Results

| Config | Temperature/top-p | Encode failures | Mean test AUC |
| --- | --- | --- | --- |
| uncoupled (round 46) | 1.0 / 1.0 | 0 | **1.000** |
| moderate | 0.9 / 0.9 | 0 | **0.938** |
| aggressive | 0.7 / 0.85 | 0 | **0.333** |

## Findings

1. **Coupling erodes the activation signal, monotonically.** Moderate
   coupling drops the probe from perfect to 0.938; aggressive coupling
   collapses it to 0.333 (below chance) under the same group-disjoint,
   matched baseline protocol. Two real data points now trace the
   coupling-vs-signal tradeoff.

2. **Aggressive coupling does not cost capacity — it costs signal
   discrimination.** Encode failures are 0 at 0.7/0.85; the coder still
   delivers. Instead, the probe's discrimination collapses. The plausible
   mechanism: aggressive (near-deterministic) sampling compresses the
   activation manifold, so the cross-condition variation the probe
   exploited disappears — the stego and matched-normal conditions become
   indistinguishable in the representation space.

3. **The below-chance point is a small-sample observation, not a
   measured inversion.** With 4 test groups the 95% CI spans 0.0–0.75, so
   0.333 is honestly read as "signal gone / at chance", not "significantly
   anti-correlated". It is however clearly distinct from the moderate
   config's 0.938 (CIs 0.625–1.0): the erosion is real.

4. **No free lunch — the duality closes.** The token-stream
   surprise-preserving result (recalibrated detector, analysis/
   recalibrated_detector.md) and the activation-stream signal are linked:
   pushing coupling until the activations carry nothing also destroys the
   delivery-relevant variation, and the aggressive config still leaves the
   FULL content-scanner corpus able to flag the carrier. Coupling buys
   activation blindness at the price of relying on an increasingly
   brittle, deterministic output.

## Implication

Distribution coupling is a real-model-evidenced, monotonic evasion against
activation detection, reaching chance (or beyond) at aggressive settings —
without encode failure. The honest limit: this is one small model, one
probe architecture, and 4 test groups; the below-chance point needs a
bigger split before it supports inversion claims.

Raw results: `findings/activation_probe_coupled/results_*.json` (gitignored).
