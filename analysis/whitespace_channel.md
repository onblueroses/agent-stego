# The whitespace-run channel: the most mundane formatting axis, unscanned

Status: offline, deterministic, fully verified (WS-1, WS-2). Produced
2026-08-14, round 58 (goal round).

## The mechanism

Bits ride in single vs double space runs: single space = 0, double = 1.
The sender canonicalizes every pure-space run to a single space, then
doubles the first K runs for the framed bit pattern. Newline runs
(paragraph breaks) are never read by either side. Double spaces are the
most common typing artifact in real documents, so a cover with some
doubled spaces is unremarkable — and no registry feature counts
regular-space runs (`nbsp_density` is about U+00A0 only).

## Measured results (offline, deterministic)

From `experiments/whitespace_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Delivery | **20/20** |
| Capacity | 1 bit per pure-space run (HELLO fits one long file) |
| Registry features at chance | **17/17** — all at exactly 0.5 |
| Catching features | **NONE** — the third fully invisible formatting axis |
| Real-tokenizer round trip (5 models) | double spaces preserved (WS-2) |

## The class, now systematic

| Axis | Channel | Registry verdict |
| --- | --- | --- |
| Letter case | CaseChannel (r63) | 17/17 blind, empty catching list |
| Date locale | DateFormatChannel (r64) | 17/17 blind, empty catching list |
| Space runs | WhitespaceChannel (r65) | 17/17 blind, empty catching list |

Three structurally different formatting axes — characters, values, and
whitespace — are all invisible to the registry. The blind spot is a
systematic class: **all-ASCII document formatting**.

## The measured class boundary

The class ends where formatting stops being ASCII:

| Formatting | non_ascii_density | Registry verdict |
| --- | --- | --- |
| Double spaces | 0.0 | invisible |
| Letter case, date locale | 0.0 | invisible |
| Curly quotes (`"`) | > 0 | **caught** |
| Em-dashes (—) | > 0 | **caught** |
| Bullet markers (•) | > 0 | **caught** |

The non_ascii_density scanner is the class boundary: any formatting that
uses non-ASCII glyphs is caught at J=1.0; all-ASCII formatting is invisible.

## Honest negatives (measured, not asserted)

1. **Every text linter flags double spaces.** This is the channel's most
   obvious tell — a lint-style consistency monitor (or spell-checker)
   catches it trivially. Unlike case and date locale, whose tells require
   a human eye or a formatting-aware monitor, double spaces are flagged by
   off-the-shelf tooling.
2. **The sender's canonicalization flattens natural double spaces**, so the
   cover must not rely on them (they become bit positions).
3. **Newlines are excluded by design** — paragraph structure is untouched,
   but a cover whose spacing is all-newline has no capacity.

## Verification

`WS-1` re-derives the 10/10 delivery, the original-17-feature
blindness, and the round-66 closure by formatting_anomaly_fraction;
`WS-2` re-verifies double-space preservation on the five real
tokenizers. See analysis/formatting_consistency_defense.md for the
defense. The full suite runs green and 42/42 claims hold on the VPS.

Raw results: `findings/whitespace/results_*.json` (gitignored).
