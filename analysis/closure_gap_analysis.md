# The closure gap analysis: the formatting defense's measured edge map

Status: offline, deterministic, fully verified (GAP-1, HSMOOTH-1).
Produced 2026-08-14, round 66 (goal round).

## The question

The formatting-consistency closure (`formatting_anomaly_fraction`, rounds
66/68) catches the case, date-format, whitespace, and heading channels at
J=1.0. Its sub-rules are PRECISION rules — each was chosen to avoid a
false-positive class. The gap hunter probes every sub-rule with a minimal
evading artifact and maps where the defense fails.

## The measured edge map

From `experiments/closure_gap_hunter.py`:

| Closure sub-rule | Evasion artifact | Closure score | Verdict |
| --- | --- | --- | --- |
| case (mid-word flips only) | word-initial flips ("Report") | **0.0** | **GAP** (proper-noun ambiguity; WORD-1) |
| heading (skips > +1) | smooth #/## alternation | **0.0** | **GAP** (normal nesting; HSMOOTH-1) |
| space (word-separating runs) | line-start indentation | **0.0** | **GAP** (excluded to avoid JSON false positives) |
| date ((formats-1)/dates) | any codebook mix | 0.35 | **gap-free** (receiver shares the codebook) |

Three precision gaps, one gap-free rule. The baselines confirm the rules
work on their intended targets: a heading skip scores 0.33 (closed),
word-separating double spaces score 0.44 (closed), mid-word flips score
positive (closed).

## Why the gaps exist

Each precision rule was a false-positive fix that created an evasion edge:

1. **case**: the rule counts mid-word flips to avoid flagging proper nouns
   ("Project Specification") — so word-initial flips are invisible, and a
   flipped word-initial is byte-identical to a proper noun (structurally
   unclosable).
2. **heading**: the rule flags level SKIPS (> +1 from the predecessor) —
   so a smooth #/## alternation (normal section/subsection nesting) never
   fires. A document with 44 sections each having one subsection is a
   plausible spec, and its outline IS the payload.
3. **space**: the rule excludes line-start runs (to avoid flagging
   indented JSON) — so a channel encoding in leading-space counts evades
   it. (This edge is the least deployable: 88 lines of varied indentation
   reads as sloppy formatting, and leading whitespace is fragile across
   tools.)
4. **date**: the rule and the receiver share the codebook, so any mix of
   the 8 formats is caught — no edge exists.

## The structural reading

The closure is a precision instrument: it closes its four axes exactly,
and every precision choice is an attack surface. Two of the three gaps are
structural (proper-noun ambiguity; normal-nesting indistinguishability) —
no syntactic rule can close them without semantic knowledge. The date rule
is the only one with no edge, precisely because it is not a precision rule:
it measures a count, not a pattern.

## The built instance: smooth 2-level heading (HSMOOTH-1)

`HeadingChannel(levels=2)` carries 1 bit per heading via #/## alternation,
invisible to all 18 features (17/17 original + the closure). The corpus
family `file_content/heading_smooth` uses a 90-heading spec cover
(45 sections, each with one subsection) — 88 bits for HELLO in one
plausible document.

## Verification

`GAP-1` re-derives the three gaps and the gap-free date rule; `HSMOOTH-1`
re-derives the 10/10 delivery and the 18-feature empty catching list for
the smooth-heading channel. The full suite runs green and 50/50 claims
hold on the VPS.

Raw results: `findings/closure_gap_hunter/results_*.json`,
`findings/corpus_matrix/results_*.json` (gitignored).
