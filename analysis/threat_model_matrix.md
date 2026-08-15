# The threat-model matrix: which channel works under which threat model

Status: operational synthesis of the 41-family corpus (53/53 claims, 492
tests). Produced 2026-08-14, round 72 (goal round). Each row names the
sender, the receiver, the monitor's observables, the deployment
requirements, and the honest viability verdict.

## The roles

- **Sender**: who controls the carrier. Framework (the agent harness),
  model (the LLM itself), router/gateway, or infrastructure.
- **Receiver**: who reads the channel. Needs the protocol (codebook/menu/
  seed) or an exhaustive scan.
- **Monitor**: the defender; its observables decide detectability.

## The matrix (selected families; the full set lives in the per-family docs)

| Family | Sender | Receiver needs | Monitor catches via | Deployment requirement | Viability |
| --- | --- | --- | --- | --- | --- |
| character carriers (zero-width, homoglyph, ...) | framework (exact content) | exhaustive text scan (protocol-free) | content density scanners | framework writes exact text | practical (content-detectable) |
| case / whitespace / heading | framework | protocol or scan | formatting closure (closed) | framework + no proper nouns / long docs | was stealthy, now closed |
| word-initial / smooth-heading | framework | protocol | nothing (structural gap) | proper-noun-free / 90-heading covers | stealthy but cover-constrained |
| date-format | framework | codebook | formatting closure (date rule gap-free) | changelog-style docs | closed |
| task-selection / retrieval / tool-codebook | framework (work/read/tool choice) | menu/codebook | call_count (volume) | agent with equivalent options | volume-limited (stealth budget) |
| correction (typo channel) | framework | codebook | call_count | framework writes wrong-then-right; final state perfect | volume-limited |
| timing / resource / length / error | framework | threshold | latency/cpu/length/error scanners + volume | framework controls the observable | modality-instrumented |
| cache (prefix-cache) | framework (request structure) | hit/miss or invoice | latency + PROVIDER (owns cache) | provider prefix caching; 3x repetition rescues churn | provider-aware |
| routing | router/gateway | model_id or style (0.5 no-log) | volume + latency + GATEWAY logs | multi-model gateway | gateway-aware |
| interleave | framework (issue order) | pairing rule | call_count; reorder kills (partially rescued) | scheduler preserves order | fragile |
| honeypot / honeypot-cache | framework | protocol (real) vs naive (decoy) | content + latency (deception, not evasion) | any content-writing agent | deception under inspection |
| inbound (poisoned file) | file author | exhaustive scan | response-aware scanners | agent reads untrusted files | practical |
| semantic (synonym) | model (capable) | synonym codebook | synonym scanner | capable sender (parked locally) | needs larger models |
| hopping / meta-hop | framework | seed / header | content + volume | multi-carrier framework | coordination |

## The five threat models, and which channels fit

1. **Content monitor only** (scans arguments/responses for carriers):
   every non-content channel evades — timing, resource, length, error,
   cache, routing, interleave, correction, task-select, retrieval,
   tool-codebook, honeypot. Caught: character carriers, formatting (closed).
2. **Content + volume monitor**: the volume-only channels (identity,
   behavioral) are caught at J=1.0; the budget is 5-30 bits per run
   (STEALTH-1).
3. **Infrastructure-aware monitor** (provider/gateway): cache and routing
   are caught — the operator owns the medium.
4. **Formatting-consistency monitor**: the four closed formatting axes are
   caught; the structural gaps (word-initial, smooth-heading) remain —
   syntactically unclosable.
5. **Semantic monitor** (knows the domain): correction (wrong values),
   semantic (synonyms), and the selection channels are caught by a
   defender who knows what "right" looks like.

## The operational verdict

- **No channel is stealthy against every monitor.** The closest are the
  structural-gap formatting channels (invisible to the current registry,
  but cover-constrained and human-visible) — and even they are caught by
  a semantic monitor.
- **The practical stealth frontier is: content channels against content
  scanners (before the closure), volume channels against everything
  except volume (within the stealth budget), infrastructure channels
  against everyone except the operator.**
- **The single most important defender capability is a semantic/ground-
  truth monitor** — it catches correction, semantic, and selection
  channels that every syntactic scanner misses.

## Verification

Every number in this matrix re-derives from the per-family claims and the
corpus matrix (CORPUS-1); the full suite runs green and 54/54 claims hold
on the VPS.
