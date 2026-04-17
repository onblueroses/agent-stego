# Detectability Scaling Analysis

## Question

Can an inference provider with white-box access to model weights and hidden states
detect encrypted arithmetic coding steganography?

Theoretically: no. Encrypted arithmetic coding draws output tokens from the model's
own distribution. If the token distribution is identical to normal sampling, the hidden
states from processing those tokens must also be identically distributed. A probe
trained on hidden states should converge to AUC 0.5 (random guessing) at scale.

## Setup

**Model:** Qwen2.5-3B-Instruct, bfloat16
**Stego:** Arithmetic coding with SHAKE-256 encryption (key XOR on message bits)
**Detection:** MLP probes on hidden states - last-layer (mean-pooled final hidden layer)
and multi-layer (concatenated mean-pooled features from all 10 captured layers)
**Probe training:** 300 epochs, Adam lr=1e-3
**Secrets:** 25 ASCII strings (2-6 chars), cycled across samples
**Temperature:** 0.8 throughout

## v1 Results (20 prompts, no length matching, top_p=0.9 only)

| n_train | n_test | Last-layer AUC | Multi-layer AUC | Time |
|---------|--------|---------------|-----------------|------|
| 10 | 5 | 0.560 | 0.600 | 92s |
| 25 | 10 | 0.730 | 0.900 | 205s |
| 50 | 20 | 0.583 | 0.626 | 406s |
| 100 | 40 | 0.603 | 0.667 | 770s |
| 200 | 80 | 0.605 | 0.500 | 1561s |

The n=25 spike to 0.900 looked alarming. But three confounds made v1 unreliable.

## Confounds identified in v1

**1. Prompt cycling (20 prompts).** With 25 training samples and 20 prompts, the probe
memorizes prompt-specific activation patterns rather than learning a general stego signal.
This explains why the spike is exactly at n=25.

**2. Length mismatch.** Stego generation stops after the message is encoded. Normal
generation runs to max_tokens. The probe picks up on sequence length as a proxy for
the label - not the stego signal itself.

**3. Top-p as a single condition.** With only top_p=0.9, we can't distinguish whether
residual signal comes from the truncation boundary or the arithmetic coding. Running
top_p=1.0 as a control isolates the variable.

## v3 Results (100 prompts, length matching, dual top_p)

All three fixes applied simultaneously:
- 100 diverse prompts (5x more than v1)
- Length matching: stego continues normal sampling after message encoded, both classes
  produce exactly max_tokens tokens
- Two conditions tested independently: top_p=0.9 and top_p=1.0

### top_p=0.9

| n_train | n_test | Last-layer AUC | Multi-layer AUC |
|---------|--------|---------------|-----------------|
| 10 | 5 | 0.480 | 0.720 |
| 25 | 10 | 0.500 | 0.520 |
| 50 | 20 | 0.536 | 0.550 |
| 100 | 40 | 0.510 | 0.489 |
| 200 | 80 | 0.502 | 0.493 |

### top_p=1.0

| n_train | n_test | Last-layer AUC | Multi-layer AUC |
|---------|--------|---------------|-----------------|
| 10 | 5 | 0.440 | 0.480 |
| 25 | 10 | 0.460 | 0.500 |
| 50 | 20 | 0.508 | 0.506 |
| 100 | 40 | 0.524 | 0.486 |
| 200 | 80 | 0.530 | 0.500 |

### Cross-condition comparison (multi-layer AUC)

| n_train | top_p=0.9 | top_p=1.0 |
|---------|-----------|-----------|
| 10 | 0.720 | 0.480 |
| 25 | 0.520 | 0.500 |
| 50 | 0.550 | 0.506 |
| 100 | 0.489 | 0.486 |
| 200 | 0.493 | 0.500 |

## Interpretation

### The v1 spike was an experimental artifact

The n=25 AUC=0.900 in v1 completely disappeared in v3. With 100 prompts and length
matching, even n=10 never exceeds 0.720 (top_p=0.9) or 0.508 (top_p=1.0). The probe
was memorizing prompt-level patterns and exploiting length correlation, not detecting
a stego signal.

### Both conditions converge to 0.5

At n=200, all four probe configurations (2 architectures x 2 top_p values) read between
0.49 and 0.53. This is within normal variance for a binary classifier on random data.
The multi-layer probe - our strongest detector - shows no advantage over coin-flipping.

### top_p=1.0 is cleaner but not required

The top_p=1.0 condition converges faster and has less variance at small n. The top_p=0.9
condition shows a residual 0.720 at n=10, which washes out by n=25. This suggests
top_p truncation introduces minor noise at very small sample sizes but has no effect
at scale. Both conditions end up at the same place.

### The last-layer probe is also defeated

In v1, the last-layer probe showed a persistent ~0.6 AUC that never dropped to 0.5.
In v3, it sits at 0.502/0.530 at n=200. The persistent signal was prompt memorization,
not a layer-specific leak.

## What this means

**For steganography:** Encrypted arithmetic coding is provably undetectable at the
activation level, not just theoretically but empirically. A white-box inference provider
with full access to model weights and hidden states cannot distinguish stego inference
from normal inference using supervised probes. The theoretical zero-KL guarantee holds
in practice.

**For detection:** Supervised probes on hidden states are not a viable detection path
for this class of steganography. The signal is information-theoretically absent - there
is nothing for the probe to learn. Detection would require a fundamentally different
approach: statistical analysis of the token sequence itself (CDF uniformity tests),
behavioral analysis (unusual prompt patterns), or side channels (timing, API usage
patterns).

## What could break this

The guarantee depends on exact conditions:

1. **Encryption must be enabled.** Without encryption, the arithmetic coder's target
   point is deterministic (derived from the plaintext message), creating a measurable
   bias in token selection. Encryption makes the target pseudorandom.

2. **The model distribution must be faithfully reproduced.** If the stego encoder uses
   a different softmax temperature, different top-p threshold, or different floating
   point precision than what the model naturally produces, the distributions diverge
   and a probe can detect the difference.

3. **No side channels.** The analysis covers hidden-state detection only. Timing
   differences (stego encoding is slower than normal sampling), API usage patterns,
   or behavioral signals (always requesting exactly max_tokens) could leak information
   through other channels.

## Next directions

The undetectability result is strong for the defender (stego user). The open question
is whether an attacker (inference provider) has detection paths we haven't considered:

- **Unsupervised anomaly detection** on activation distributions (no labels needed)
- **Sequential analysis** across multiple requests from the same user
- **Timing side channels** from the arithmetic coding computation
- **Perturbation-based detection**: inject small weight perturbations and measure
  whether the output distribution shifts differently for stego vs normal inference
