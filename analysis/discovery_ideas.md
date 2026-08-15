# Discovery ideas: wild, interesting, and useful steganography

Status: ideation card for the rewritten discovery objective (2026-08-14).
Inspired by the elder-plinius (Pliny the Elder) corpus — ST3GG's breadth,
GLOSSOPETRAE's tokenizer-blindness and semantic stego, P4RS3LT0NGV3's
concealment catalog, T3MP3ST's reproducibility — but each idea below is meant
to extend or go beyond what those repos demonstrate. Tagged by
offline-testability: `offline` (deterministic, no model/API), `hybrid`
(executor-simulated sender, deterministic monitors), `model` (needs an LLM /
API budget, parked until approved).

## Candidate mechanisms

1. **Polyglot document carriers** (`offline`). **DONE as the duplicate-key
   parser differential** (`src/channels/polyglot.py`, `analysis/polyglot_documents.md`):
   bits in JSON duplicate-key resolution — first-wins/canonicalizing readers
   see a normal config, byte-preserving readers recover the payload; any
   re-serialization collapses it. The multi-format variant (JSON+Markdown,
   YAML+Python) remains open.

2. **Error-channel revival** (`hybrid`). The historical error channel
   (success/failure bit patterns) was capability-gated noise on small models
   but showed signal on Qwen 3.6. An executor that simulates an exact
   fail/succeed controller + the detector suite would measure its detection
   profile (intentional failures are statistically visible) — the honest
   question is *how* visible.

3. **Tool-order permutation across turns** (`offline`). Round-4 hopping hops
   carriers; the mirror image is hopping *tools*: the order of tool calls
   within a turn (Lehmer-coded, like the historical permutation channel) with
   a seed-derived per-turn schedule. Extends the coordination theme to the
   surface class.

4. **Agent-memory / state carrier** (`hybrid`). **DONE**
   (`src/channels/numberfmt.py`, `analysis/number_format_carrier.md`):
   number-format spellings (`1000` vs `1e3`, `1.5` vs `1.50`, `0` vs `-0`)
   carry bits in state/config files; canonicalization is the complete
   defense; the spelling scanner isolates the axis at J=1.

5. **Semantic stego, offline-simulated** (`hybrid`). **DONE**
   (`src/channels/semantic.py`, `analysis/semantic_synonym_choice.md`):
   synonym-choice carrier with a third orthogonal axis (lexical);
   canonicalizing monitor blind, synonym-surface scanner J=1, character
   and diversity scanners blind. The real version needs a capable sender
   model (0.5B/0.6B failed to execute linguistic encoding in the overnight
   P2 run).

6. **Response-length channel** (`hybrid`). The historical response_length
   channel: the model controls its prose length before each tool call, and
   length bins encode bits. An executor that emits length-tagged responses
   + a length-histogram detector would measure its profile (the harness's
   ToolTrace already records token_count).

7. **Prompt-injection-carried stego** (`model`). ST3GG's agent section: hide
   *instructions* (not just data) in carriers — a file whose hidden payload
   changes what the agent does next. The wildest agent-native use case; needs
   real models to demonstrate behavioral effects.

8. **Cross-agent relay, rebuilt** (`hybrid`). The historical relay demo: one
   agent encodes while appearing to work; a second agent with only the raw
   trace log independently discovers and decodes. Rebuildable offline with
   the self-describing headers (round 5) + exhaustive decode as the "relay"
   — the discovery story (does the relay find it without the key?).

9. **Carrier meta-channel** (`offline`). A header that *itself* selects the
   carrier (round 5 extended): the first chunk declares which carrier the
   rest use, so a single conversation can switch mechanisms mid-stream
   without a shared seed — self-describing hopping.

10. **Multi-stage cross-surface composition** (`offline`). **DONE**
    (`src/channels/multistage.py`, `analysis/multistage_composition.md`):
    one payload split across tool-choice, file-content characters, and
    key-order surfaces; joint decode exact, every per-surface monitor blind,
    wrong stage orders fail closed. Extends matryoshka (round 7) from one
    document to one conversation.

## Selection criteria

- Offline-testable first (no budget), hybrid second.
- Prefer mechanisms with a *novel blindness axis* or a *coordination
  primitive* over parameter variations of existing carriers.
- Every prototype must come with a detector evaluation (what catches it) and
  a verify-claims entry — an interesting channel without a measured detection
  profile is half a finding.

## Session rounds 54-62: beyond the original list

The queue above was largely cleared and exceeded. New mechanisms built,
measured, and verified (claim ids, all offline/deterministic unless noted):

| Round | Mechanism | Claim | Blindness axis |
| --- | --- | --- | --- |
| 54 | Honeypot: real payload under a louder decoy | HONEYPOT-1 | naive monitor is handed the decoy 20/20 |
| 55 | Prefix-cache: bits in cache hit/miss state | CACHE-1, CACHE-2 | content scanners at 0.5; provider cache is the medium; real-tokenizer verified 5/5 |
| 56 | Honeypot x cache: real payload exists only as cache state | HONEYCACHE-1 | thorough content scan finds the real payload 0/20 |
| 57 | Task-selection: payload in which reports get written | TASKSEL-1 | 10 bits/turn, 2.5 bits/call, 10 content features at 0.5 |
| 58 | Multi-model routing: payload in which model answers | ROUTE-1 | 2 bits/request; no-log receiver 0.5 vs 0.25 chance (real models) |
| 59 | Concurrency interleaving: payload in merged call order | INTERLEAVE-1 | 13/13 features blind; fails closed under scheduler reorder |
| 60 | Self-correction: payload in the wrong value, erased by the fix | CORRECT-1 | final state perfect 20/20; 13/13 blind |
| 61 | Corpus matrix capstone: 34 families x 17 features | CORPUS-1 | the whole taxonomy's measured detection profile in one table |
| 62 | Retrieval-selection: payload in which document is read | RETRIEVE-1 | RAG-native inbound selection; surface mimics honest auditors |
| 63 | Case channel: payload in letter case | CASE-1, CASE-2 | 17/17 registry features at 0.5 — empty catching list; survives all 5 real tokenizers |
| 64 | Date-format channel: payload in which locale date format | DATE-1, DATE-2 | second fully invisible family (17/17 at 0.5) — the formatting-axis blind spot is a CLASS; scanner boundary measured (hyphen/dot dates flagged, slash/month-name clean) |
| 65 | Whitespace-run channel: payload in single vs double spaces | WS-1, WS-2 | third fully invisible family (17/17 at 0.5); measured boundary: the class covers all-ASCII formatting only — non-ASCII formatting (curly quotes, em-dashes, bullets) is caught by non_ascii_density |
| 66 | Formatting-consistency DEFENSE: formatting_anomaly_fraction closes the class | FORMAT-1 | case/date/whitespace caught at J=1.0 by one additive feature; false-positive-free on clean and character carriers — the blind spot is closed, not just documented |
| 67 | Heading-level channel: payload in markdown heading depths | HEAD-1, HEAD-2 | 4th formatting axis (document STRUCTURE); invisible to all 18 features for one round — a measured honest gap in the closure |
| 68 | Closure extended to heading depth | FORMAT-1 (updated) | heading-depth consistency component catches the 4th axis at J=1.0 — the formatting class is complete: every discovered axis (case/date/space/structure) has a defense; binary_tool_choice is the only no-signature family |
| 69 | Relay discovery (cross-agent #8): zero-protocol receiver | RELAY-1, RELAY-2 | 13 text-carrier families protocol-free discoverable; 26 protocol-keyed; the unkeyed relay is deceived by the honeypot (cross-validates HONEYPOT-1) |
| 70 | Word-initial-case channel: the closure's precision gap | WORD-1, WORD-2 | invisible to all 18 features — the closure counts only mid-word flips (proper-noun precision), so word-initial flips evade it; the gap is SYNTACTICALLY UNCLOSABLE (a flipped "Report" is a proper noun); covers must be proper-noun-free |
| 71 | Closure gap analysis + smooth-heading channel | GAP-1, HSMOOTH-1 | the edge map: 3 precision gaps (word-initial case, smooth #/## alternation, line-start indentation), 1 gap-free rule (date, codebook-shared); smooth-heading built as an invisible channel (normal nesting never skips) |
| 72 | Redundancy coding | REDUND-1 | repetition + majority voting: cache fragility noise-limited (r3 -> 100% at adversarial churn), interleave partially structural (r5 -> 88% at 5% reorder, still fail-closed at 25%) |
| 73 | Tool-codebook channel | TOOLCB-1 | bits in WHICH of 4 equivalent tools works (2 bits/call); volume-only — completes the identity-codebook family (model/work/doc/tool) |
| 74 | Instruction stego pilot | measured negative | 0/4 models follow an invisible zero-width behavioral instruction — invisible carriers carry data (INBOUND-1) but not behavior at <=3.8B scale |

Residual queued items (final status, round 72):

- prompt-injection-carried *instruction* stego (`model`, MEASURED-NEGATIVE
  at local scale: `instruction_stego_pilot.py` — 0/4 models follow an
  invisible zero-width behavioral instruction; invisible text carries
  data (INBOUND-1) but not behavior at <=3.8B scale; a capable
  instruction-tuned sender remains the only unmeasured case).
- semantic stego with capable senders (`model`, parked — the synonym
  carrier is offline-proven; real senders need larger models or API
  budget; P2 measured small models at 0/16 on linguistic encoding).
- tool-order hopping (subsumed by task-selection and interleave).
- error revival (superseded by the self-correction channel).
- cross-model relay (DONE: relay discovery RELAY-1/RELAY-2 plus the
  historical relay_receiver_capability measurements).

The offline taxonomy is saturated: character, structural, lexical,
serialization, coordination, behavioral, agent-native, formatting,
infrastructure, and defense-analysis families are all built, measured, and
claimed. The remaining frontier is real-model capability (measured weak at
<=3.8B scale) and any future formatting axes the registry adds (the class
loop).
