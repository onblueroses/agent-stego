# The discovery program at round 72: synthesis

Status: consolidation of rounds 54-79 (goal rounds 54-72) of the rewritten
discovery objective, all numbers re-derivable (54/54 claims, 492 tests).
Produced 2026-08-14.

## The arc in one paragraph

The session extended the elder-plinius corpus in five waves. First,
**deception**: the honeypot (real payload under a louder decoy). Second,
**infrastructure channels**: bits in the provider's prefix-cache state and
in the model router's choice — payloads with no textual existence at all.
Third, **selection channels**: bits in which legitimate work happens
(task-selection, retrieval-selection). Fourth, the **behavioral/agent
family**: concurrency interleaving, self-correcting typos. Fifth, the
**formatting class**: case, date-locale, space-run width, and heading depth
— four axes wholly invisible to the original registry, then closed by one
consistency feature. Throughout: honest negatives measured, not asserted,
and every headline re-derived by verify-claims.

## New mechanisms (rounds 54-69), with claims

| Mechanism | Claim(s) | The one-line finding |
| --- | --- | --- |
| Honeypot (real under a louder decoy) | HONEYPOT-1 | naive monitor is handed the decoy 20/20 |
| Prefix-cache side channel | CACHE-1, CACHE-2 | payload exists only as cache hit/miss state; the invoice decodes it; mechanism verified on 5 real tokenizers |
| Honeypot x cache composition | HONEYCACHE-1 | real payload exists NOWHERE in content (0/20 thorough scans) |
| Task-selection | TASKSEL-1 | bits in which legitimate reports get written (2.5 bits/call) |
| Multi-model routing | ROUTE-1 | bits in which model answers (2 bits/request); no-log receiver 0.5 vs 0.25 chance |
| Concurrency interleaving | INTERLEAVE-1 | bits in the merged call order; fail-closed under scheduler reorder |
| Self-correction (typo channel) | CORRECT-1 | bits in the wrong value, erased by the fix; real-model execution 4/12 |
| Retrieval-selection | RETRIEVE-1 | bits in which document is read (RAG-native, 3 bits/retrieval) |
| Case channel | CASE-1, CASE-2 | bits in letter case; first fully-invisible family |
| Date-format channel | DATE-1, DATE-2 | bits in which locale date format (3 bits/date) |
| Whitespace channel | WS-1, WS-2 | bits in single vs double spaces |
| Heading-level channel | HEAD-1, HEAD-2 | bits in markdown heading depth |
| Formatting-consistency defense | FORMAT-1 | one additive feature closes all four formatting axes |
| Word-initial-case channel | WORD-1, WORD-2 | the closure's precision gap: invisible to all 18 features, and syntactically unclosable (a flipped word-initial is a proper noun) |
| Relay discovery | RELAY-1, RELAY-2 | zero-protocol relay finds only the 13 text carriers; and it is deceived by the honeypot |

## The measured honest negatives (the scientific spine)

1. **Every discovery family has a measured detection profile** — the
   corpus matrix (38 families x 18 features) shows what catches each; no
   family is asserted stealthy without the matrix row.
2. **The formatting class was invisible, then closed.** Case, date, space,
   and heading were each fully invisible to the original 17-feature
   registry; `formatting_anomaly_fraction` closes all four at J=1.0,
   false-positive-free. The one-round heading gap is documented as the
   honest proof that "a consistency feature closes only the axes it
   defines."
3. **Infrastructure channels are caught by the operator.** The provider
   owns the cache and the route — cache/routing are invisible to content
   guards, not to the infrastructure (which also has activations).
4. **Capability gates are measured, not assumed.** Small models: acrostics
   0/16 (P2), correction execution 4/12, routing style-ID 0.5 vs 0.25.
5. **The relay ceiling is a design property.** Protocol-keyed channels are
   invisible to an unkeyed relay — stealth from detectors and keyed-ness
   are the same coin.

## The corpus at a glance

- 41 channel families x 18 registry features (the matrix).
- 54 verify-claims entries, 492 tests, all green on the VPS.
- Three families have no decisive registry signature: `binary_tool_choice`
  (stealth via near-zero capacity) and the two structural precision-gap
  channels (word-initial case, smooth-heading — syntactically unclosable).
- Operational guidance: see analysis/threat_model_matrix.md (the five
  threat models and which channels fit each).

## The remaining frontier

- **Real-model generation of the formatting/selection channels**: the
  offline executors assume framework-level control; small-model execution
  is capability-gated (measured weak). A capable sender would unlock
  end-to-end demos of the case/date/heading channels.
- **Prompt-injection-carried instruction stego** (queued, parked): needs a
  capable agent loop; the offline inbound baseline exists.
- **Semantic stego with capable senders** (queued, parked): the synonym
  carrier is offline-proven; real senders need larger models or API budget.
- **Further formatting axes** (list markers, quote style): the class loop
  (discover -> close) continues wherever the registry has gaps.

## Reading order

`analysis/corpus_matrix.md` (the detection capstone) -> `analysis/
detection_duality_summary.md` (the real-model duality) ->
`analysis/formatting_consistency_defense.md` (the closure) ->
`analysis/relay_discovery.md` (the relay lens) -> `analysis/
discovery_ideas.md` (the tracking table with claim ids).
