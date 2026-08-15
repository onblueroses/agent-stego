# Tool-trace detection: results and honest interpretation

Status: offline simulation, not release-qualified findings. Produced
2026-08-13 on a private VPS runner that is not part of this release. The corpus
is synthetic — traces are emitted by the deterministic `HonestExecutor`, not
by real models — so every number below measures the *channel contracts* against
deterministic detectors, not real-world detectability.

## What was built

- `src/channels/detector.py` — feature extraction over `ToolTrace` sequences
  and a registry of single-feature threshold detectors. Features split into
  **content** (invisible-char density, NBSP density, homoglyph density,
  combining-mark density, non-ASCII density — need argument access) and
  **surface** (call count, unique tools, single-tool fraction, mean latency —
  tool names/counts/timing only).
- `src/channels/corpus.py` — labelled synthetic corpus: clean traces (report
  writers: read-write-read; auditors: list-then-read) and stego traces from
  every cross-surface composition family plus the historical table channel.
  Observations within a family are distinct (covers, messages, clocks vary).
- Calibration is honest: thresholds are fit on **clean traces only**
  (mean + 5σ; tool diversity flags *below* the clean minimum), then applied
  to unseen stego traces. AUC is reported threshold-free.

## The matrix (Youden's J at the calibrated threshold; AUC in the JSON)

| Family | Best detector | J |
| --- | --- | --- |
| file_content/zero_width | invisible-char density | 1.00 |
| file_content/nbsp | NBSP density | 1.00 |
| file_content/nfd | combining-mark density | 1.00 |
| file_content/tag_char | invisible-char density | 1.00 |
| file_content/homoglyph | homoglyph density | 1.00 |
| file_content/variation_selector | invisible-char density | 1.00 |
| argument_carrier | homoglyph density | 1.00 |
| table/path | unique tools | 1.00 |
| config/json_key_order | key-order anomaly | 1.00 |
| hopping/seeded (naive) | unique tools | 1.00 |

Every carrier is caught at J = 1.00 by a content scanner that reads argument
strings — the scanner that catches zero-width also catches tag chars and
variation selectors, because all three are non-ASCII invisible characters.
The JSON key-order family (round 3, `src/channels/json_order.py`) is caught
only by its structural detector: decoys restore the call shape (unique-tools
J = 0.0) and the carrier is ASCII (invisible-char J = 0.0), isolating the
mechanism orthogonally. The seeded hopping family (round 4,
`src/channels/hopping.py`) is a coordination channel, not an invisibility
one: the naive (write-only) version is caught by tool diversity at J = 1.0,
and a full-registry content scanner catches every run (12/12, FPR 0) while
single-carrier density detectors are partially diluted (J = 0.75). See
`analysis/hopping_channels.md`.

## The arms-race contrast (the interesting part)

Adding **decoy reads** to the file-content prompt (read two project files
first, then write) restores the natural call pattern. The tool-diversity
scan is the tell:

| Family | unique-tools J |
| --- | --- |
| file_content/zero_width **naive** (single write call) | 1.00 |
| file_content/zero_width **with decoys** (read-write-read) | 0.00 (AUC 0.5) |
| content scanner on the decoyed version | 1.00 |

Interpretation:

1. **Naive composition is trivially surface-detectable.** Every current
   channel prompt confines the agent to one tool (write-only or read-only),
   while natural agents mix tools. A diversity scan catches all of them at
   J = 1.00. This contradicts any "near-undetectable" intuition for the
   naive versions in this corpus.
2. **Decoys defeat surface scanners.** Matching the cover task's call shape
   drops the diversity detector to chance (AUC 0.5). This is the
   cover-task-fidelity requirement the charter demands be measured.
3. **Content carriers remain detectable by any content-reading scanner.**
   A monitor with argument access catches every carrier at J = 1.00. The
   only configuration where content carriers are invisible to a monitor is
   the tokenizer-blindness asymmetry from the tripartite harness (round 1):
   a monitor whose tokenizer strips the carrier before it reads the
   content. Content scanners with a preserving tokenizer always win.

So the measured frontier for these compositions is: surface detection is
defeated by decoys; content detection is defeated only by tokenizer
blindness; a preserving content-reading monitor is a complete defense.

## Known limitations

- Synthetic corpus, deterministic executors; no real-model variance.
- Threshold calibration on 24 clean traces; k=5 is a policy choice, not a
  measured operating point.
- The table channel here is the path mode via an exact-copy executor; real
  models deviate (historically the reason table channels had nonzero CER).
- These are contract-level measurements, not deployment claims.

Raw results: `findings/trace_detection/results_*.json` (gitignored).
