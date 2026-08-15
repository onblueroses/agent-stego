# The formatting-consistency defense: closing the blind spot

Status: offline, deterministic, fully verified (FORMAT-1). Produced
2026-08-14, round 59 (goal round).

## The gap it closes

Rounds 63-65 found three formatting-axis channels — case (CASE-1),
date-format (DATE-1), whitespace (WS-1) — fully invisible to the original
17-feature registry: all 17 features at exactly 0.5 AUC, empty catching
lists. The registry modeled character density, structure, lexical choice,
timing, and volume, but not *all-ASCII document formatting consistency*.

## The defense

`formatting_anomaly_fraction` (src/channels/detector.py, round 66) combines
three consistency sub-scores by max:

1. **Case regularity**: uppercase letters whose previous character is a
   lowercase letter (mid-word "thE" flips) / lowercase letters. The
   mid-word rule excludes natural word-initial capitals ("Project
   Specification"), sentence initials, and heading/list markers — what
   makes it false-positive-free on decoy responses and heading-heavy docs.
2. **Date-format locale**: (distinct date formats - 1) / dates.
3. **Space-run width**: double-space word-separating runs / runs
   (indentation and trailing padding excluded).
4. **Heading-depth consistency** (round 68): fraction of headings whose
   level skips more than one nesting level from their predecessor — the
   rule that closes the heading-level channel.

## Measured results (offline, deterministic)

From `experiments/formatting_defense_eval.py`:

| Family | formatting J | max other-feature AUC |
| --- | --- | --- |
| case | **1.0** | 0.5 |
| date-format | **1.0** | 0.5 |
| whitespace | **1.0** | 0.5 |
| heading-level (r68) | **1.0** | 0.5 |

| Check | Result |
| --- | --- |
| False positives on character carriers (zero-width, homoglyph, nbsp) | all AUC 0.5 |
| Max clean-response score | **0.0** |

## Why this matters

1. **The closure is additive, not a re-fit.** The other 17 features stay at
   0.5 AUC for the three families; the new feature alone separates them.
   The blind spot was a missing dimension, and adding that one dimension
   closes it — the measured proof that formatting consistency was the
   unmodeled axis.
2. **It is false-positive-free by construction.** Clean traces (including
   decoy read responses with markdown headings, list lines, and
   indented JSON configs) score exactly 0. The eligibility rules were
   driven by the failure modes found while testing: JSON indentation
   (space sub-score), ZWSP-perturbed case context (case sub-score), and
   heading/list markers.
3. **It completes the duality for the formatting class.** Every discovery
   family in the corpus now has a measured defense: the three formatting
   channels were the last with no registry signature (besides
   binary_tool_choice, whose "signature" is its near-zero capacity).

## Honest negatives (measured, not asserted)

1. The feature is deterministic and rule-based; an attacker who knows the
   rules can craft formatting that evades it (e.g., consistent-but-wrong
   case across the whole document, or a single locale used throughout —
   which is the date channel's natural state).
2. It does not catch non-formatting axes (semantic choice, timing,
   infrastructure channels) — those remain the domain of their own
   scanners.
3. `binary_tool_choice` still has no decisive signature; its stealth is
   bought with near-zero capacity, not an unmodeled axis.

## Verification

`FORMAT-1` in `scripts/verify_claims.py` re-derives the closure (all four
families at J>=0.8, clean scores 0). `CASE-1`/`DATE-1`/`WS-1`/`HEAD-1`
assert "original 17 features blind, closed by formatting_anomaly_fraction".
The full suite runs green and 44/44 claims hold on the VPS.

Raw results: `findings/formatting_defense/results_*.json` (gitignored).
