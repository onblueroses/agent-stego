# The honeypot channel: a real payload under a louder decoy

Status: offline, deterministic, fully verified (HONEYPOT-1). Produced
2026-08-14, round 54.

## The mechanism

The artifact carries TWO framed payloads on the same cover:

1. the **real message** in homoglyphs (subtle: ASCII-visible substitutions,
   low density);
2. a **decoy** in zero-width characters (loud: invisible characters, the
   single most salient carrier in the corpus).

A monitor that runs the exhaustive carrier registry and stops at the first
framed payload finds the zero-width decoy first (registry order) and reads
`DECOY-PAYLOAD`. Only a receiver that knows the protocol — strip the decoy,
decode homoglyph — recovers the real message.

## Measured results (offline, deterministic)

From `experiments/honeypot_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Protocol-aware real-payload recovery | **20/20** |
| Naive monitor (first framed payload) recovers the decoy | **20/20** |
| Naive monitor recovers the real payload | **0/20** |
| Thorough monitor (all carriers scanned) finds both | 20/20 |

## Honest negatives (measured, not asserted)

1. **A thorough monitor wins.** Scanning every carrier (not stopping at the
   first hit) recovers both payloads on every artifact. The honeypot
   deceives a monitor with a stopping rule; it does not hide from a
   systematic one.
2. **The decoy adds no marginal detectability.** Against the corpus
   detector, both the honeypot/decoy family and the homoglyph-only baseline
   score AUROC 1.000 (vs clean). The real carrier is already fully
   detectable in this corpus; the honeypot's value is **deception under
   inspection, not evasion**.
3. **Detectability is set by the decoy by design.** Any defender whose
   detector fires on invisible characters will flag the trace — and will
   then be handed the decoy. The trade is explicit: higher salience in
   exchange for a false flag.

## Why this is interesting

The channel inverts the usual stego priority. Every other carrier in the
corpus minimizes its footprint; the honeypot deliberately maximizes one
footprint to absorb the monitor's attention. It is a *deniability* channel:
when caught, the attacker's story ("that's just the decoy text") is
indistinguishable from the truth, because the decoy IS what the monitor
finds. It also composes: the same trick could put the real payload in a
carrier the monitor never scans (e.g., response-length, timing) while the
zero-width decoy absorbs a content scan.

## Verification

`HONEYPOT-1` in `scripts/verify_claims.py` re-derives the 10/10 decoy-first
and 10/10 real-recovery numbers from the library code; the full suite runs
green (403 tests) and 26/26 claims hold on the VPS.

Raw results: `findings/honeypot/results_*.json` (gitignored).
