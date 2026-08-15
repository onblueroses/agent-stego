# The prefix-cache side channel: the provider's cache is the medium

Status: offline, deterministic, fully verified (CACHE-1). Produced
2026-08-14, round 55.

## The mechanism

LLM inference providers cache token prefixes between requests (vLLM-style
block caches; OpenAI, Anthropic, Gemini, DeepSeek all expose cache-hit
latency and often cache-aware pricing). A request that reuses the previous
request's prompt prefix hits the cache (fast); one with a benign comment
inserted at the start of the prompt misses (slow). The bits ride in the
hit/miss sequence — a robust binary observable — while the tool calls and
their arguments stay ordinary.

The channel is state-emergent, not sender-controlled: the sender does NOT
set the latency; it emerges from the shared cache state. The receiver needs
only hit/miss, not precise timing — a hit at 0.1s vs a miss at 1.0s is a
two-order-of-magnitude, threshold-robust distinction. Bit 1 reuses the
stable system prefix; bit 0 prepends a UNIQUE index-stamped comment (a
repeated comment would itself become cached and turn consecutive 0s into
hits). The comment is content-ordinary — counters in comment lines are
normal in tool logs.

## Measured results (offline, deterministic)

From `experiments/cache_side_eval.py`:

| Metric | Result |
| --- | --- |
| Delivery without interference | **20/20** |
| BER vs cache churn (p per request) | p≤0.5 → **0.0**; p=1.0 (burst after every request) → **0.74** |
| BER vs cache capacity | ≥16 blocks (~256 tokens) → **0.0**; 8 blocks → 0.8; ≤4 → 1.0 |
| Content-feature AUCs (10 scanners) | **all 0.5** — perfectly blind |
| Latency-variance monitor | catches at **J = 1.0** |
| Billing decode (cache-aware pricing 1x vs 4x) | **recovered from the invoice** |

## Why this is wild

1. **The infrastructure is the medium.** Sender and receiver share state
   through the provider's cache without either writing payload into visible
   content. The payload never appears in any tool argument, file, or
   response — it exists only as a latency difference caused by cache state.

2. **The invoice is the side channel.** Providers price cached input tokens
   below uncached ones. A receiver with the provider's billing statement
   and a known prefix length decodes the hit/miss sequence from the bill —
   no timing access needed. The cost differential is the payload.

3. **It is the only channel in the corpus that is invisible to every
   content scanner by construction.** All ten content/structural/lexical
   features sit at exactly 0.5 AUC. The only detectable signature is the
   latency variance itself — which is the channel.

4. **It composes with the honeypot.** The decoy-absorbing trick from round
   54 can sit on the content side while the real payload rides the cache —
   a monitor that flags the zero-width decoy never even looks at timing.

## Honest negatives (measured, not asserted)

1. **The provider sees everything.** The provider owns the cache and can
   read the hit/miss pattern directly — this channel is invisible to
   third-party content guards, not to the inference provider (who, per the
   detection-duality arc, also has activation access).
2. **Sustained cache churn degrades it — and repetition FULLY rescues
   it (REDUND-1).** An adversarial tenant evicting after every request
   drives delivery to 15% (r1); 3x repetition with majority voting
   restores 100% — the fragility was noise-limited, not structural.
3. **Small caches fail.** Below ~16 blocks of capacity the session's own
   pollution evicts the working set and the channel dies (BER 1.0).
4. **Volume signature.** Framing needs one request per bit (~48 requests
   for "HELLO"), so call-count scanners catch the volume.
5. **Simulation caveat, now partially retired.** The simulator tokenizes
   with whitespace-delimited units. CACHE-2 verifies the MECHANISM against
   the real tokenizers of all five local models: identical-prefix reuse
   tokenizes identically (hit), nonce comments break the first 16-token
   block (miss), and consecutive nonces stay distinct at the block level —
   on every vendor. One honest subtlety surfaced: the leading "#" token is
   SHARED across nonces; the nonce digit lands inside the first block,
   which is what separates consecutive 0-requests. What remains
   simulator-specific is the exact latency/BER curve, which depends on the
   provider's cache implementation (block size, eviction policy, tenancy),
   not on tokenization.

## Verification

`CACHE-1` re-derives the 10/10 delivery and `CACHE-2` re-verifies the
mechanism against the real tokenizers of all five local models (reuse
hits, nonce misses, consecutive 0s distinct) from the library code; the
full suite runs green and 34/34 claims hold on the VPS.

Raw results: `findings/cache_side/results_*.json` (gitignored).
