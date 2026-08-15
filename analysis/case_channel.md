# The case channel: the most visible property nobody scans for

Status: offline, deterministic, fully verified (CASE-1, CASE-2). Produced
2026-08-14, round 56 (goal round).

## The mechanism

The payload rides in the CASE of mid-sentence letters: uppercase = 1,
lowercase = 0. The sender canonicalizes every eligible (non-sentence-
initial) letter to lowercase, then sets the first K eligible letters to
the framed bit pattern. Sentence-initial capitals are preserved and never
read (both sides share the eligibility rule: a letter is ineligible if it
starts the text or follows '.!?').

The channel lives entirely inside one ordinary report file — a single
write, so it carries no volume, timing, or diversity signature.

## Measured results (offline, deterministic)

From `experiments/case_channel_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Delivery | **20/20** |
| Capacity | 1 bit per eligible letter (HELLO fits one file) |
| ORIGINAL registry features at chance | **17/17** — all at exactly 0.5 |
| Catching features (original registry) | **NONE** — closed by formatting_anomaly_fraction (r66) at J=1.0 |
| Exhaustive monitor (scans all carriers) | recovers the payload (honest) |
| Real-tokenizer round trip (5 models) | byte-identical; payload survives (CASE-2) |

## Why this is wild

1. **The most visible typographic property is the least monitored.** Case is
   the first thing a human notices in text — yet no feature in the detector
   registry models case regularity. Every density scanner (invisible chars,
   homoglyphs, non-ASCII), every structural scanner, every lexical,
   diversity, latency, and volume feature sits at exactly 0.5 AUC. This is
   the first content family with an EMPTY catching list — even
   binary_tool_choice had a weak volume signal; the case channel has none.
2. **The payload is in the SAME letter that renders it.** Unlike zero-width
   or homoglyph carriers (which add or substitute characters), the case
   channel changes a letter's presentation only — the word is still the
   word, the text is still the text, the file is still a genuine report.
3. **It survives every real tokenizer we have** (CASE-2): round trips are
   byte-identical on Qwen2.5/Qwen3/gpt2/Phi, so delivery to a real model
   preserves the pattern exactly.

## Honest negatives (measured, not asserted)

1. **The pattern is visible to humans and case-aware monitors.** The
   channel is invisible to the REGISTRY, not to a reader who notices
   mid-sentence capitals, and not to any monitor that checks case
   regularity. Its stealth is against density/structure scanners, which is
   exactly the class every other content carrier is caught by.
2. **The canonicalized cover is typographically flat.** Sentence-initial
   capitals survive, but mid-sentence normalization reads slightly
   mechanical to a careful human.
3. **An exhaustive monitor finds it.** The channel is registered, so a
   monitor that scans all known carriers recovers the payload — the
   shared-registry protocol is its vulnerability, like every carrier.
4. **Eligibility is a shared rule.** Both sides must agree on the
   sentence-initial definition; a monitor that models case against a
   different grammar could desynchronize a naive receiver (the sender and
   receiver in this repo share the exact rule).

## Real-model generation feasibility (measured)

`experiments/case_copy_feasibility.py` asked the four local models to copy
a case-encoded text verbatim. Result: **exact copy 0/12** (no model
reproduces text byte-for-byte — small models paraphrase, consistent with
P2's weak-executor finding), but **the payload survives 1/12**:
Phi-3.5-mini reproduced the case-flipped pattern — including the STX1
frame — before drifting into paraphrase ("ThE sTatUS iS hEaLthy AnD The
depLOymeNt is ready...").

Two honest readings:

1. **Delivery is framework-level at this scale.** The FileContentChannel
   sender (which writes the exact content) is the realistic deployment;
   asking the MODEL to produce the pattern works only intermittently on
   the strongest local model (3.8B), and not at all at <=1.5B.
2. **The framing is partially-copy-robust.** The payload survives even a
   model that copies the pattern then drifts, because the frame sits in
   the first ~48 bits — a promising property for a capable sender.

## Verification

`CASE-1` re-derives the 10/10 delivery, the original-17-feature
blindness, and the round-66 closure by formatting_anomaly_fraction;
`CASE-2` re-verifies the byte-identical round trip on the five real
tokenizers. See analysis/formatting_consistency_defense.md for the
defense. The full suite runs green and 42/42 claims hold on the VPS.

Raw results: `findings/case_channel/results_*.json`,
`findings/case_real_tokenizers/results_*.json` (gitignored).
