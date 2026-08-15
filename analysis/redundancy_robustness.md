# Redundancy coding: rescuing the fragile channels

Status: offline, deterministic, fully verified (REDUND-1). Produced
2026-08-14, round 68 (goal round).

## The question

Two session findings measured fragility: the interleave channel is
fail-closed under scheduler reordering (0% delivery at 5% per-pair
reorder), and the cache channel degrades under cross-client churn (15%
delivery at adversarial p=1.0 interference). Both failures are FRAME
failures: one corrupted bit anywhere in the 88-bit frame kills decode.
This experiment asks whether the generic rescue layer — R-fold repetition
of every bit with majority voting at the receiver — recovers delivery, and
where it stops working.

## Measured results (offline, deterministic)

From `experiments/redundancy_robustness.py` (60 trials per cell):

**Cache: delivery rate vs interference probability**

| Repetition | p=0.0 | p=0.5 | p=1.0 (adversarial) |
| --- | --- | --- | --- |
| r1 | 1.0 | 1.0 | **0.15** |
| r3 | 1.0 | 1.0 | **1.0** |
| r5 | 1.0 | 1.0 | 1.0 |

**Interleave: delivery rate vs per-pair reorder probability**

| Repetition | 0% | 5% | 10% | 25% |
| --- | --- | --- | --- | --- |
| r1 | 1.0 | **0.02** | 0.0 | 0.0 |
| r3 | 1.0 | 0.53 | 0.08 | 0.0 |
| r5 | 1.0 | **0.88** | 0.50 | 0.0 |

## The findings

1. **The cache fragility was noise-limited, not structural.** Three votes
   per bit take the channel from 15% to **100% delivery** even under
   adversarial churn (an interfering burst after every single request).
   The channel's hit/miss margin is large enough that majority voting
   recovers every bit.
2. **The interleave fragility is partially structural.** Repetition
   rescues the low-noise regime (5x: 2% -> 88% at 5% reorder, 0% -> 50%
   at 10%), but the channel still fails closed at 25% reorder: majority
   voting over 5 copies cannot overcome a per-bit error rate that high on
   an 88-bit frame. The order channel's fragility is real, not just
   frame-induced — unlike the cache channel.
3. **Repetition is a generic, channel-agnostic layer.** The same wrapper
   (repeat bits, run the executor, majority-vote) works on both channels
   with zero changes to the channels themselves. Cost: R-fold volume —
   the honest price for robustness.

## Honest negatives (measured, not asserted)

1. Repetition multiplies the volume signature (R x calls), making
   call-count detection strictly easier — robustness and stealth trade
   off directly.
2. The interleave channel's 25%-reorder regime is beyond majority voting;
   an error-correcting code (not just repetition) would be the next step,
   at further volume cost.
3. The cache rescue assumes the interference is the ONLY noise; real
   providers add latency jitter that would erode the hit/miss margin.

## Verification

`REDUND-1` re-derives the four headline cells (cache r1 fragile / r3
rescued; interleave r1 fail-closed / r5 rescued) from the library code;
the full suite runs green and 51/51 claims hold on the VPS.

Raw results: `findings/redundancy_robustness/results_*.json` (gitignored).
