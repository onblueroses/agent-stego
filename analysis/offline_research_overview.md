# Offline research overview: mechanism taxonomy, measured matrix, and honest limits

Status: synthesis of the offline simulation program (rounds 1-9), not
release-qualified findings. Every number below is recomputed from code by
`scripts/verify_claims.py` (run on the VPS); none is copied from prose. All
traces and documents are produced by deterministic executors — no real model
has been run, and every tokenizer-asymmetry assumption is a published research
claim (GLOSSOPETRAE E3S) pending local measurement.

## The arc

| Round | Added | Mechanism class |
| --- | --- | --- |
| 1 | 6 text carriers + tripartite harness + file-content composition | character |
| 2 | tool-trace detector suite + decoy arms race | defense |
| 3 | JSON key-order carrier | structural |
| 4 | seeded multi-carrier hopping | coordination |
| 5 | self-describing trace headers | protocol |
| 6 | verify-claims command | integrity |
| 7 | matryoshka documents (structural + character) | capacity |
| 8 | multi-turn file-content exfiltration | protocol |
| 9 | historical channel coverage (volume, binary) + this overview | coverage |
| 10 | polyglot duplicate-key parser differential | parser differential |
| 11 | multi-stage cross-surface composition | surface fragmentation |

## Mechanism taxonomy

Three carrier classes, each with its own blindness axis:

| Class | Carriers | Blindness axis | Defense that defeats it |
| --- | --- | --- | --- |
| Character | zero-width, tag-char, NBSP, NFD, homoglyph, variation-selector | tokenizer | a tokenizer that strips/normalizes the carrier |
| Structural | JSON key order | canonicalization | `sort_keys=True` re-serialization |
| Surface/coordination | tool selection, call counts, schedules, headers | observation scope | a monitor restricted to tool names/counts |

Protocol layers compose the carriers: single-document (file-content,
key-order, matryoshka), multi-document (hopping), multi-turn
(multi-turn exfiltration), and self-describing (headers).

## The measured detection matrix (15 families)

Every family is caught at J = 1.00 by at least one detector in the 5-sigma
registry, **except one documented gap**:

| Family | Best detector | J |
| --- | --- | --- |
| file_content/{6 carriers} | respective content scanner | 1.00 |
| argument_carrier | homoglyph density | 1.00 |
| table/path | unique tools | 1.00 |
| config/json_key_order | key-order anomaly | 1.00 |
| config/matryoshka | key-order / invisible-char | 1.00 |
| config/duplicate_keys | duplicate-key fraction | 1.00 |
| text/semantic_synonym | synonym-surface fraction | 1.00 |
| multistage/cross_surface | call count / content scanners | 1.00 |
| timing/inter_call | latency-std / call count | 1.00 |
| resource/cpu_side | cpu-std / call count | 1.00 |
| error/pattern | error-fraction / call count | 1.00 |
| length/response | token-count-std / call count | 1.00 |
| inbound/poisoned_file | invisible-char (response-aware) | 1.00 |
| spread/multicarrier | invisible + nbsp + homoglyph (multiplied) | 1.00 |
| multiturn/file_content | invisible-char density | 1.00 |
| volume | call count | 1.00 |
| hopping/seeded | unique tools (naive) | 1.00 |
| **binary_tool_choice (HI, 10 calls)** | **none** | **0.00** |

The binary-tool-choice gap is length-dependent: at 5 bits/character, "HI"
(10 calls) sits under the call-count threshold with two tools and no content
signal, but "SEARCH" (30 calls) is caught at J = 1.00. The registry's blind
spot is short, low-count, low-diversity traces — a real, measured limitation.

## The honest negatives (what does not work)

1. **Naive single-tool traces are trivially detectable.** Every prompt that
   confines the agent to one tool is caught by tool diversity at J = 1.00.
   Decoy calls that match the cover task's shape restore surface stealth
   (diversity J drops to 0.00) — cover-task fidelity is a requirement, not
   an option.
2. **Content carriers are statistically visible.** Any content-reading
   scanner catches every character/structural carrier at J = 1.00. The only
   blindness is a monitor whose own preprocessing (tokenizer stripping,
   canonicalization) destroys the carrier.
3. **Coordination is not invisibility.** Hopping and self-describing headers
   change who can decode and how much it costs to search, but they do not
   hide carrier presence from a full-registry scanner (hopping: single-carrier
   J drops to 0.75, full-registry catch 12/12).
4. **Bandwidth is the price of robustness.** Self-framing and position
   metadata cost: multi-turn exfiltration is 72 bits/character; hopping pays
   a 48-bit frame plus rotation. The historical table channel's ~5 bits/char
   remains the capacity champion precisely because it carries no framing.

## What is claimed vs hypothesis

- Measured (deterministic, offline): every number in `verify_claims.py`.
- Hypotheses (published claims, unverified locally): tokenizer asymmetry
  tables (GLOSSOPETRAE E3S), real-model instruction-following fidelity, and
  anything about actual LLM behavior. The executors simulate exact
  prompt-following, which real models approximate imperfectly — the
  historical results showed exactly where that breaks (early stopping,
  off-by-one counts).

## The detection duality, measured (round 46)

The token-likelihood coder is **output-level invisible** (surprise-preserving:
mean normalized surprise 0.99 = native expectation; CDF-uniform by design) but
**activation-level trivial** (group-disjoint probe AUC 1.000 x3 seeds). The
historical invalidated duality table is restored in corrected form — and
moderate distribution coupling (0.9/0.9) erodes the probe to AUC 0.938,
the first measured evasion data point.

## The open frontier — now partly measured

The overnight real-model suite (rounds 10-20, `analysis/overnight_measurement_report.md`)
replaced part of the simulation with facts: measured open-tokenizer asymmetry
(character carriers preserved; NFD the only blind spot), a real
semantic-stego delivery x catch matrix (small models cannot execute acrostics;
LLM monitors cannot perceive character carriers), real token-stego capacity
(4/4 exact, 3-7 bits/token with a measured perplexity frontier), an *inverted*
output-level detector on real data, and a zero-call tool-capability floor at
0.5B. Semantic stego with capable models and real cross-vendor tripartite runs
still need an API budget.

Per-round details: `analysis/external_research/elder_plinius_ecosystem.md`,
`analysis/trace_detection_results.md`, `analysis/json_document_channels.md`,
`analysis/hopping_channels.md`, `analysis/self_describing_traces.md`,
`analysis/matryoshka_documents.md`, `analysis/multiturn_exfiltration.md`.
Raw results: `findings/**/results_*.json` (gitignored).
