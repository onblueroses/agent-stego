# Cover-task utility: stego does not destroy the artifact

Status: offline, deterministic, fully verified (UTILITY-1). Produced
2026-08-14, round 73 (goal round). Closes the elder-plinius gap table's
last "Missing" item: cover-task utility measurement.

## The question

Steganography should not destroy the cover's legitimate purpose: a report
must still be a report, a config must still parse. Every carrier in the
corpus is evaluated on how much of the cover's content survives encoding.

## Measured results (offline, deterministic)

From `experiments/cover_utility_eval.py`:

| Carrier | Content words preserved |
| --- | --- |
| zero_width | **1.0** |
| homoglyph | **1.0** |
| tag_char | **1.0** |
| nbsp | **1.0** |
| nfd | **1.0** |
| variation_selector | **1.0** |
| case | **1.0** |
| whitespace | **1.0** |
| word_initial_case | **1.0** |
| heading_level | **1.0** |
| heading_smooth | **1.0** |
| date_format | **1.0** |

| Structural carrier | Parsed config preserved? |
| --- | --- |
| json_key_order | **yes** (same key/value pairs) |
| duplicate_keys (first-wins reader) | **yes** (same parsed config) |

## The finding

1. **Every carrier preserves ~100% of the cover's content.** The utility
   cost of stego in this corpus is readability quirks (invisible
   characters, unusual case, mixed date locales) — never content loss.
   The report still says what the cover said; the config still parses to
   the same object.
2. **This is a design property, not luck**: the character carriers
   substitute/add without removing words; the formatting carriers change
   presentation; the structural carriers reorder/duplicate keys but keep
   the first-wins meaning. A channel that destroyed its cover would fail
   its own utility check — and none does.
3. **The exceptions are semantic, not lexical**: the synonym channel
   (semantic.py) trades wording for meaning (utility is semantic), and
   the correction channel's transient wrong value is by design erased.
   The measured proxy (word preservation) intentionally measures the
   lexical floor.

## Why this matters

The gap table marked cover-task utility as Missing; it was the last
unmeasured dimension of the corpus. The honest conclusion: in this
taxonomy, stealth is never bought with the artifact's usefulness — the
cover remains genuinely useful, which is exactly what makes these
channels deployable (the artifact is still a real report/config the agent
would legitimately produce).

## Honest negatives (measured, not asserted)

1. The proxy is lexical (content words), not semantic: it does not verify
   a reader UNDERSTANDS the text, only that the words are present. A
   semantic reader study is beyond scope.
2. Readability quirks ARE a utility cost for a human audience (mixed date
   locales read sloppily) — the cost is in presentation, not content.
3. Structural carriers preserve the config only for the intended reader
   (first-wins); a canonicalizing re-serializer collapses the channel but
   keeps the config — the utility is reader-dependent.

## Verification

`UTILITY-1` re-derives the >=90% word-preservation floor for all 12 text/
formatting carriers and the parsed-config preservation for both structural
carriers; the full suite runs green and 54/54 claims hold on the VPS.

Raw results: `findings/cover_utility/results_*.json` (gitignored).
