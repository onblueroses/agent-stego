# The date-format channel: locale variation is the payload

Status: offline, deterministic, fully verified (DATE-1, DATE-2). Produced
2026-08-14, round 57 (goal round).

## The mechanism

Changelogs, datelines, and logs legitimately carry many dates, and date
formats legitimately vary by locale. The channel uses an 8-format codebook
(slash-ISO, US slash, "14 Aug 2026", "Aug 14, 2026", "Aug 14 2026",
compact, full-month variants); each date in a document carries 3 bits by
which of the 8 formats renders it. Every format is a REAL locale
convention, so no single date is anomalous — only the MIX is the payload.

## Measured results (offline, deterministic)

From `experiments/datefmt_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Delivery | **20/20** |
| Capacity | **3 bits per date** (8-format codebook); HELLO needs ~30 dates |
| Registry features at chance | **17/17** — all at exactly 0.5 |
| Catching features | **NONE** — the second fully invisible family |
| Real-tokenizer round trip (5 models) | byte-identical; payload survives (DATE-2) |

## Why this is wild

1. **It proves the blind spot is a CLASS, not a channel.** The case channel
   (round 63) showed one formatting axis (case regularity) is unmodeled.
   The date channel shows a SECOND, structurally different formatting axis
   (date-format consistency) is equally unmodeled — both have completely
   empty registry catching lists. The registry models character density,
   structure, lexical choice, timing, and volume, but not *document
   formatting consistency*: a whole class of typographic/locale axes is
   invisible to it.
2. **The payload is locale variation.** A changelog mixing "2026/08/14",
   "Aug 14, 2026", and "20260814" is a document written in multiple locale
   conventions — each rendering individually legitimate, the combination
   meaningful only to the receiver.
3. **Changelogs are the natural carrier.** Datelines and release notes
   carry many dates by convention, so the volume needed for a frame
   (30 dates) is ordinary document structure, not an anomaly.

## The measured scanner boundary (a finding in itself)

The registry's number-format scanner (`number_format_anomaly_fraction`)
does generalize to date strings, but only through their digit runs:
- `2026-08-14` (hyphen, leading-zero month) → the `-0`-style negative
  token is flagged anomalous;
- `01.10.2026` (dot, `dd.mm` ending in 0) → flagged;
- `14.08.2026` (dot, not ending in 0) → **clean**;
- slash, month-name, and compact renderings → **clean** (pure integer
  digit runs).

The codebook deliberately uses only the clean renderings; including a
hyphen or risky dot date makes the family detectable at J=1.0. This is the
measured edge of the number-format scanner: it catches leading-zero and
trailing-zero number tokens, not date formats per se.

## Honest negatives (measured, not asserted)

1. **A locale-consistency monitor sees the mix.** The channel is invisible
   to the REGISTRY, not to a human or a monitor that checks date-format
   consistency (docs should use one locale). Mixed formats read as sloppy
   writing.
2. **The codebook is the shared protocol.** The receiver must know the
   format-to-bits mapping — a keyed scheme.
3. **The scanner boundary constrains the codebook.** Real conventions
   using hyphens (ISO) or dots (European) are excluded because the
   number-format scanner flags their digit runs; a stronger attacker uses
   only the clean renderings.
4. **An exhaustive text-channel scan does not find it** — it is not a
   character carrier; detection requires format-aware inspection.

## Verification

`DATE-1` re-derives the 10/10 delivery, the original-17-feature
blindness, and the round-66 closure by formatting_anomaly_fraction;
`DATE-2` re-verifies the byte-identical round trip on the five real
tokenizers. See analysis/formatting_consistency_defense.md for the
defense. The full suite runs green and 42/42 claims hold on the VPS.

Raw results: `findings/date_format/results_*.json`,
`findings/date_format_real/results_*.json` (gitignored).
