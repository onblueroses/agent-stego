# The word-initial-case channel: the closure's precision gap

Status: offline, deterministic, fully verified (WORD-1, WORD-2). Produced
2026-08-14, round 64 (goal round).

## The mechanism

Bits ride in the case of word-initial letters: lowercase = 0, uppercase =
1. The sender canonicalizes every eligible word-initial letter (a letter
that starts a word AND is not sentence-initial) to lowercase, then sets
the first K to the framed pattern. Sentence-initial capitals are preserved
and never read (shared rule: skip letters that start the text or follow
'.!?').

This looks like the case channel (round 63) — except it flips only the
FIRST letter of words, which is exactly the one case position the
formatting closure does not measure.

## The finding: the closure's precision is its weakness

The round-66/68 closure's case sub-score counts only **mid-word** flips
(uppercase preceded by a lowercase letter — the "thE" pattern). That rule
was chosen to avoid false positives on natural word-initial capitals
("Project Specification"). The flip side is a structural gap: a channel
that flips only word-initial letters scores **0.0 on the closure** and is
invisible to all 18 registry features (WORD-1: 17/17 original + the
closure, all at exactly 0.5).

And the gap is **syntactically unclosable**: a flipped word-initial
"Report" is byte-identical to a proper noun "Report". The closure scores
BOTH zero (measured in the tests) — no rule based on characters, spacing,
or structure can separate the channel from natural proper-noun text.
Closing it requires semantic knowledge (is this a noun? is it a name?),
which a deterministic registry does not have.

## Measured results (offline, deterministic)

| Metric | Result |
| --- | --- |
| Delivery | **20/20** (10/10 in WORD-1) |
| Capacity | 1 bit per eligible word-initial (HELLO fits one long file) |
| Registry features at chance | **18/18** — including the formatting closure |
| Catching features | **NONE** |
| Closure on flipped text vs proper-noun text | both **0.0** — indistinguishable (the unclosability proof) |
| Real-tokenizer round trip (5 models) | byte-identical; payload survives (WORD-2) |

## Why this matters

1. **Defenses have precision/recall tradeoffs, and this is the measured
   attack side of one.** The closure chose precision (exclude proper nouns)
   and paid with a gap. The word-initial channel is the exploit of that
   choice — invisible to everything, including the dedicated defense.
2. **The gap is fundamental, not a missing rule.** No syntactic feature
   can distinguish a flipped "Report" from a proper noun "Report"; the
   attacker pays the same price (proper nouns in the cover would be
   misread as bits, so covers must be proper-noun-free — the channel's
   honest constraint).
3. **The class loop continues honestly**: each closure spawns a precision
   gap, and each gap is either closable (heading depth was) or structural
   (this one is). The corpus now documents both kinds.

## Honest negatives (measured, not asserted)

1. **The cover must be proper-noun-free.** Any natural word-initial
   capital in the cover would be misread as a bit — the receiver cannot
   distinguish. The channel's canonicalization handles the sender's side;
   the constraint is on cover selection.
2. **Human-visible weirdness**: random word-initial capitals read as
   title-case or emphasis — visible to a human reader, like the case
   channel's tell.
3. **A semantic monitor** (one that knows which words are proper nouns)
   could separate the channel — the unclosability is syntactic, not
   semantic.
4. **Exhaustive text scan finds it** (it is a registered carrier): the
   zero-protocol relay would recover the payload (protocol-free, like the
   other character carriers).

## Verification

`WORD-1` re-derives the 10/10 delivery, the 18-feature empty catching
list, and the unclosability proof (closure scores 0 on both flipped and
proper-noun text); `WORD-2` re-verifies the byte-identical round trip on
the five real tokenizers. The full suite runs green and 48/48 claims hold
on the VPS.

Raw results: `findings/corpus_matrix/results_*.json` (gitignored).
