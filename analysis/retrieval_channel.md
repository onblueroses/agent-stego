# The retrieval-selection channel: which document is the payload

Status: offline, deterministic, fully verified (RETRIEVE-1). Produced
2026-08-14, round 56 (goal round 56).

## The mechanism

RAG agents query a retrieval index and read the returned document. The
retrieval SELECTION is a covert channel: the index (or query phrasing) is
controlled so each retrieval returns one of eight near-identical status
documents (`docs/status_0.md` .. `status_7.md`), and the returned
document's identity carries 3 bits. The receiver reads which document was
retrieved and unframes.

The selected document is then genuinely used — the agent's work is informed
by it — so there is no stego artifact: the document is a real status report
and the read is a real read.

## Measured results (offline, deterministic)

From `experiments/retrieval_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Delivery | **20/20** |
| Capacity | **3 bits per retrieval** (8-doc codebook) |
| Registry features at chance | **13/13** (content, diversity, latency) |
| call-count monitor | catches at **J = 1.0** (read volume) |
| Reads per run (HELLO) | ~30 vs clean auditors' 5 |

## Why this is wild

1. **The inbound mirror of task-selection.** Task-selection (round 57)
   writes the payload into WHICH reports are produced; retrieval-selection
   puts it in WHICH document the agent READS. The surface mimics the honest
   auditor exactly (list_files + reads) — the same tool names, the same
   shapes — and the retrieved document is genuinely consumed.
2. **RAG is the most common agent-native retrieval pattern.** A poisoned
   index or a query-steering exploit makes the framework itself the
   sender: every agent that retrieves and reads is a potential carrier, and
   the retrieved document's identity is observable to anyone with the
   trace.
3. **Capacity without content.** 3 bits per read, zero payload material in
   any document — the codebook of near-identical documents is a pure
   selection axis, like the task menu but inbound.

## Honest negatives (measured, not asserted)

1. **Read volume is the tell.** ~30 reads for "HELLO" vs clean auditors'
   5; call-count scanners separate the family at J=1.0.
2. **The shared codebook is the protocol.** The receiver must know the
   doc-identity-to-bits mapping — a keyed scheme.
3. **Real deployment needs retrieval control.** The sender must influence
   which document the index returns (poisoning or query steering); the
   offline simulator assumes that control. Without it the channel is not
   operable.
4. **A monitor that models the doc-set distribution** (which documents
   exist, how often each is read) sees the same selection the receiver
   reads.

## Verification

`RETRIEVE-1` in `scripts/verify_claims.py` re-derives the 10/10 delivery
from the library code; the full suite runs green and 35/35 claims hold on
the VPS.

Raw results: `findings/retrieval/results_*.json` (gitignored).
