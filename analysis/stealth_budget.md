# The stealth budget: how many bits before the volume signature trips

Status: offline, deterministic, fully verified (STEALTH-1). Produced
2026-08-14, round 71 (goal round).

## The question

The corpus matrix proved volume (call_count) is the universal tell for
bit-per-call channels. This measures the OPERATIONAL limit: how many bits
can each volume-only family deliver while staying UNDER the calibrated
call-count threshold (clean mean + 5 sigma)?

## The measured budget

From `experiments/stealth_budget.py`: the calibrated threshold is **11.2
calls** (24 clean runs, mean + 5 sigma), so a run may make at most **11
calls** before flagging.

| Family | Bits/call | Payload calls | **Budget** |
| --- | --- | --- | --- |
| interleave | 0.5 | 10 | **5 bits** |
| timing | 1.0 | 10 | 10 bits |
| correction | 1.5 | 10 | 15 bits |
| task_selection | 2.0 | 10 | 20 bits |
| routing | 2.0 | 10 | 20 bits |
| tool_codebook | 2.0 | 11 | 22 bits |
| retrieval | 3.0 | 10 | **30 bits** |

The honest answer: **volume-only channels can deliver 5-30 bits (under 4
bytes) per run before the call-count signature trips.**

## The content-channel contrast

Content-carrier features (density scanners like invisible_char_density)
have threshold exactly **0.0**: ANY carrier character is caught. So
content channels have essentially **zero stealth budget** — but they can
deliver the full 88-bit frame in ONE file write (no volume tax at all).

This is the quantified capacity-stealth tradeoff:

| Channel class | Stealth budget | Capacity per artifact |
| --- | --- | --- |
| Content (formatting, characters) | ~0 bits (any carrier caught) | 88+ bits in one file |
| Volume (identity, behavioral) | 5-30 bits before flagging | unbounded (but detected) |
| Structural gap (word-initial, smooth-heading) | unbounded (invisible) | constrained by cover requirements |

## Why this matters

1. **The volume tax is now priced.** The matrix said "volume is the
   universal tell"; this says *how much* you can transmit before the tell
   fires: a few bytes. Any exfiltration beyond that requires either a
   single-artifact carrier (content) or an invisible axis (structural
   gaps).
2. **The retrieval channel is the most efficient volume-only carrier** (3
   bits/call -> 30-bit budget) — its per-call density is the highest in
   the volume-caught group.
3. **It reframes the leaderboard**: the "stealthiest" channels are those
   whose budget is bounded by cover constraints (structural gaps) or by
   the frame itself (content, where the budget is ~0 but the frame fits
   one artifact), not by the volume threshold.

## Honest negatives (measured, not asserted)

1. The budget assumes the run starts at the clean baseline; a busy agent
   (more legitimate calls) raises the threshold and the budget — the
   budget is a floor, not a ceiling.
2. The threshold is the mean+5 sigma calibration of THIS corpus; a
   defender with a tighter calibration (lower k) shrinks the budget.
3. Repetition coding (REDUND-1) multiplies calls and therefore halves
   the budget per useful bit — robustness and stealth trade off
   directly.

## Verification

`STEALTH-1` re-derives the threshold range, the retrieval budget, and the
zero content-feature threshold from the library code; the full suite runs
green and 53/53 claims hold on the VPS.

Raw results: `findings/stealth_budget/results_*.json` (gitignored).
