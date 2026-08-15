# Real activation-probe detection: the duality, measured correctly

Status: real-model pilot on the VPS (Qwen2.5-0.5B-Instruct, CPU), not
release-qualified findings. Produced 2026-08-14.

## The question

The round-1 detection-duality table historically reported activation-probe
AUC ~1.0, later invalidated by train/test prompt-group leakage. This
experiment re-measures it with the corrected protocol: a 2-layer MLP probe
trained on hidden states, **group-disjoint split** (whole prompt groups,
never repeated observations), **matched normal baseline** (temperature 1.0,
top-p 1.0 sampling — the baseline that surprise detection could not
separate), three seeds, bootstrap CIs.

## Results

| Seed | Held-out test AUC (4 prompt groups) |
| --- | --- |
| 0 | 1.000 (CI [1.0, 1.0]) |
| 1 | 1.000 (CI [1.0, 1.0]) |
| 2 | 1.000 (CI [1.0, 1.0]) |

Mean test AUC = **1.000**.

## Findings

1. **The detection duality is real and now measured correctly.** Outside the
   provider, the coder is output-level invisible (round 41: mean normalized
   surprise 0.99 = native expectation; CDF-uniform by design). Inside the
   provider, activations separate perfectly: AUC 1.000 under the
   leakage-free protocol that the historical claim was invalidated for. The
   two halves of the duality now both have corrected, real-model evidence.

2. **The coder is surprise-preserving but not activation-preserving.** The
   arithmetic coding changes which tokens are selected; the hidden state
   sequence reflects those selections, and a mean-pooled all-layer feature
   separates stego from matched sampling trivially. The indistinguishability
   claim holds only at the statistics a text-level observer can compute.

3. **Honest scope.** n_test=4 prompt groups (small but group-disjoint — the
   exact fix for the historical leakage), 0.5B model, mean-pooled features,
   120-epoch MLP. The clean 1.0/1.0 CIs make the separation unambiguous in
   this configuration; larger models and per-position features are natural
   follow-ups.

## Implication for defense

An inference provider with hidden-state access can detect this coder
perfectly; an external observer cannot (by surprise/CDF statistics). The
defensive tier that matters is the provider — matching the historical
"access tiers" framing, now with corrected evidence on both sides.

Raw results: `findings/activation_probe_real/results_*.json` (gitignored).
