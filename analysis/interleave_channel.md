# The concurrency-interleaving channel: the merged timeline is the payload

Status: offline, deterministic, fully verified (INTERLEAVE-1). Produced
2026-08-14, round 59.

## The mechanism

Multi-agent frameworks run parallel work streams whose tool calls merge
into one observable timeline. The interleaving ORDER of two streams is a
covert channel: in each pair of calls (one from stream A = read_file, one
from stream B = search_web) the sender issues A-then-B for bit 1 and
B-then-A for bit 0. The receiver chunks the merged timeline into
consecutive pairs and reads which stream went first.

Every call is ordinary; inter-call gaps are uniform (0.3s), so timing
carries nothing. The ORDER is the only signal — and no feature in the
corpus detector models intra-pair order.

## Measured results (offline, deterministic)

From `experiments/interleave_eval.py`:

| Metric | Result |
| --- | --- |
| Delivery under a faithful scheduler | **20/20** |
| Capacity | 1 bit per pair (0.5 bits/call) |
| Registry features at chance | **13/13** (content, latency, diversity) |
| call-count monitor | catches at **J = 1.0** (2 calls per bit) |
| Delivery vs scheduler reorder ("HELLO", 88 pairs) | 1.0 at 0%; 0.42 at 1%; **0.0 at 5%** |
| Delivery vs scheduler reorder ("HI", 48 pairs) | 1.0 at 0%; ~0.05 at 5% |

## Why this is wild

1. **Concurrency is the last unmonitored dimension.** Content scanners,
   structure scanners, latency monitors, diversity checks — none of the
   corpus's 13 non-volume features separates this family (all exactly 0.5
   AUC). The intra-pair order of parallel work is a dimension that simply
   is not modeled by existing detectors, because nobody thinks of the
   scheduler's merged timeline as a payload carrier.
2. **Two legitimate streams, zero stego artifacts.** The calls are two
   ordinary parallel tasks (read config, search status); there is no
   payload material anywhere — only the order in which interleaved calls
   landed.
3. **It is the agent-framework analog of a race-condition channel.** The
   payload exploits the framework's own scheduling observable, the same
   surface that race conditions and TOCTOU bugs live on — a genuinely
   infrastructure-native mechanism the plinius corpus never touches.

## Honest negatives (measured, not asserted)

1. **A faithful scheduler is required — but redundancy partially
   rescues it (REDUND-1).** The frame is fail-closed: at 1% per-pair
   reorder "HELLO" delivers 42%; at 5%, ~2%. Repetition coding with
   majority voting takes 5% reorder from ~2% to ~88% delivery (5x
   repetition) — yet the channel still fails closed at 25% reorder, so
   its fragility is PARTIALLY structural, not purely frame-induced.
2. **The merged-timeline order IS the channel.** An order-aware monitor
   that models call sequencing sees the same pattern the receiver reads —
   the channel and its detection axis coincide.
3. **Volume is extreme**: 2 calls per bit; call-count scanners separate
   the family at J=1.0.
4. **The pairing rule and stream identity are the protocol.** The receiver
   must know the chunk-of-two grouping and which tool belongs to which
   stream — a shared-protocol scheme like every keyed channel.

## Verification

`INTERLEAVE-1` in `scripts/verify_claims.py` re-derives the 10/10 delivery
from the library code; the full suite runs green and 31/31 claims hold on
the VPS.

Raw results: `findings/interleave/results_*.json` (gitignored).
