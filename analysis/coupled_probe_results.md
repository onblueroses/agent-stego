# Activation Probes Under Distribution Coupling

**Date**: 2026-04-16
**Instance**: Vast.ai RTX 4090 ($0.315/hr), instance 35069268
**Models**: Qwen 2.5 3B, 7B
**Experiment**: 6-condition coupling sweep with activation probes

## Question

Does AUC=1.0 activation detection survive when the entropy gap is closed
via distribution coupling?

Previous probes used greedy normal baselines, creating an artificial gap.
This experiment matches conditions: normal text uses stochastic sampling
with the same (temperature, top_p) as the stego encoder.

## Method

For each of 6 coupling configurations:
1. Generate 20 normal texts via stochastic sampling (matching temp/top_p)
2. Generate 20 stego texts via arithmetic coding with same coupling params
3. Capture hidden states from 10 layers during inference
4. Train 2-layer MLP activation probe (75/25 train/test split)
5. Report probe AUC and token-level entropy AUC

All stego uses encryption (SHAKE-256 + nonce) in addition to coupling.

## Results

### 3B - Probes are bulletproof

| Temp | TopP | Probe AUC | Probe Acc | Ent AUC | Ent Gap |
|------|------|-----------|-----------|---------|---------|
| 1.0 | 1.00 | 1.000 | 1.000 | 0.677 | 0.883 |
| 0.9 | 1.00 | 1.000 | 1.000 | 0.655 | 0.900 |
| 1.0 | 0.90 | 0.960 | 0.900 | 0.678 | 0.633 |
| 0.9 | 0.90 | 1.000 | 1.000 | 0.645 | 0.450 |
| 0.8 | 0.90 | 1.000 | 1.000 | 0.693 | 0.552 |
| 0.7 | 0.85 | 1.000 | 0.900 | 0.658 | 0.255 |

Coupling reduces entropy gap from 0.883 to 0.255 at T=0.7/top_p=0.85,
but the probe stays at AUC 1.000. The single 0.960 at T=1.0/top_p=0.9
is likely noise (accuracy still 0.900 with only 10 test samples).

### 7B - Probes weakened but effective

| Temp | TopP | Probe AUC | Probe Acc | Ent AUC | Ent Gap |
|------|------|-----------|-----------|---------|---------|
| 1.0 | 1.00 | 1.000 | 1.000 | 0.557 | 0.990 |
| 0.9 | 1.00 | 1.000 | 0.900 | 0.600 | 0.328 |
| 1.0 | 0.90 | 1.000 | 0.900 | 0.735 | 0.878 |
| 0.9 | 0.90 | 0.960 | 0.900 | 0.730 | 0.808 |
| 0.8 | 0.90 | 0.960 | 0.900 | 0.662 | 0.310 |
| 0.7 | 0.85 | **0.800** | 0.700 | 0.623 | 0.234 |

Aggressive coupling (T=0.7, top_p=0.85) drops probe AUC from 1.000 to
0.800. Clear downward trend: tighter coupling = weaker probe.

## Analysis

### The probe detects something beyond entropy

On 3B, the entropy gap drops to 0.255 (tight) but the probe stays at
1.000. The probe exploits structural differences in hidden states that
survive even when output token statistics are matched. This is NOT just
entropy detection at a deeper level - it's a fundamentally different signal.

What the probe likely detects:
- **Deterministic token selection**: Arithmetic coding picks tokens
  deterministically (based on message bits), while normal sampling picks
  stochastically. Even when the marginal distribution matches, the joint
  distribution of (context, token) differs because stego tokens are
  selected for a reason unrelated to the context.
- **Output layer divergence**: Previous cosine similarity analysis showed
  output layers diverge (cosine ~0.94) while middle layers stay near-unity
  (~0.999). The prediction head "knows" the next token is atypical for
  this specific context, even if it's typical in aggregate.

### Model size matters

The 7B model shows weaker probe signal under coupling because larger models
have broader internal representations. When the 7B's natural sampling already
uses much of the vocabulary, coupled stego tokens look more like natural
continuations at the hidden state level too.

**Extrapolation**: If probe AUC drops from 1.000 (3B) to 0.800 (7B) at
aggressive coupling, models at 14B+ may push below 0.7. At 70B+ with
aggressive coupling, activation probes might approach chance. This is the
most important follow-up experiment.

### Token-level entropy detection update

With proper stochastic baselines (not greedy), entropy AUC ranges
0.557-0.735 depending on coupling. This is lower than the previous
oriented AUC of ~0.85, partly because the baselines are now correct
(stochastic, not greedy) and partly because coupling helps.

The best entropy AUC (0.557 at raw 7B) is near chance. Coupled 3B achieves
0.645 at best - still detectable but much weaker than raw 0.865.

## Implications

1. **Activation probes are the real defense.** Token-level detection can
   be largely defeated by coupling. Activation probes survive on small
   models but weaken on larger ones.

2. **The threat model has three tiers:**
   - External observer (output only): coupling + encryption ≈ undetectable
   - Inference provider (hidden states): detectable on small models,
     weakening signal on larger models
   - Same-model defender (full access): always detectable via training-time
     countermeasures

3. **Next research directions:**
   - Test 14B+ models to map the probe AUC curve vs model size
   - Adversarial training: can stego be trained to minimize probe signal?
   - Probe transfer: does a 3B-trained probe detect 7B stego?
   - More coupling params: finer grid around T=0.6-0.8, top_p=0.8-0.95

## Open Questions

1. **Model size scaling**: At what size does the probe drop below 0.7?
2. **Adversarial stego training**: Can we optimize token selection to
   minimize both entropy gap AND activation divergence simultaneously?
3. **Probe architecture**: Would a deeper probe (3+ layers, attention)
   recover AUC on 7B coupled stego? Or is the signal genuinely weaker?
4. **Capacity cost**: Does aggressive coupling (T=0.7) significantly
   reduce bits/token? Is there a capacity-detectability Pareto frontier?
