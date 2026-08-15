# The routing channel: which model answers is the payload

Status: offline, deterministic, fully verified (ROUTE-1); real-model
receiver-view measured 2026-08-14. Produced 2026-08-14, round 58.

## The mechanism

Agent gateways route each request to one of several models (priority,
fallback, cost routing). The router's choice is a covert channel: with N
models it carries log2(N) bits per request, and the chosen model's output
is ordinary — it is a real model doing real work. Nothing in any argument,
file, or response carries payload material; the payload is the ROUTE
DECISION itself.

The receiver has two views:
- **Gateway-log view**: the trace carries model_id metadata (the offline
  simulator's view, measured here).
- **No-log view**: the receiver identifies the model from the response's
  characteristic style/latency — measured on the real local models by
  `experiments/routing_discrimination.py` (leave-one-prompt-out
  nearest-centroid over a priori text features).

## Measured results (offline, deterministic)

From `experiments/routing_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Delivery (gateway-log view) | **20/20** |
| Capacity | **2 bits per routed request** (4 models) |
| Content-feature AUCs (11 scanners) | **all 0.5** — completely blind |
| call-count monitor | catches at **J = 1.0** (honest volume) |
| With cache hit/miss composed | **3 bits per request** on the same calls |
| No-log receiver (real models, 4-way style ID) | **0.5** vs 0.25 chance — partial |

## Why this is wild

1. **The infrastructure's own load-balancing decision is the channel.**
   Model routing is everywhere in production agent gateways; a compromised
   router (or a compromised sender influencing routing) exfiltrates
   through a decision the operator already makes on every request. The
   plinius corpus's carriers are all content- or tokenizer-level; nothing
   touches the multi-model routing surface.
2. **The chosen model's output is genuinely legitimate.** Unlike every
   content channel, there is no stego artifact at all — a real model
   produced real work. Content inspection is structurally blind (all 11
   features at exactly 0.5).
3. **It composes orthogonally with every other channel.** The route picks
   the model; the model's output can itself carry a character, semantic,
   or cache-pattern payload: routing (2 bits) + cache hit/miss (1 bit) =
   3 bits per request on the same tool calls.

## Honest negatives (measured, not asserted)

1. **The infrastructure operator sees the routes.** Anyone with gateway
   telemetry reads the model_id sequence directly — the channel is
   invisible to content monitors, not to the operator (who also owns the
   cache and the activations, per the detection-duality arc).
2. **The shared model menu is the protocol.** The receiver must know the
   menu (which models, which order) — a keyed scheme like every other.
3. **Volume signature.** One routed request per 2 bits (~44 requests for
   "HELLO"); call-count scanners separate the family at J=1.0.
4. **No-log viability is partial, measured, and signal-robust.** A
   style receiver (a priori text features, leave-one-prompt-out nearest
   centroid) identifies the routed model on 0.5 of held-out prompts vs
   0.25 chance; a v2 receiver adding tokenizer-fingerprint features
   (token counts under all five public tokenizers) scores the SAME 0.5 —
   the ~2x-chance limit holds across signals on these models. The raw
   outputs show why: gpt2's unsteered output is stylistically distinctive
   (code fragments), but the three instruct models produce similar
   coherent answers. Gateway logs remain the reliable receiver view; a
   no-log receiver works for coarse discrimination (instruct vs base)
   but confuses similar families.

## Verification

`ROUTE-1` in `scripts/verify_claims.py` re-derives the 10/10 delivery from
the library code; the full suite runs green and 30/30 claims hold on the
VPS.

Raw results: `findings/routing/results_*.json`,
`findings/routing_discrimination/results_*.json` (gitignored).
