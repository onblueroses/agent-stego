# The tool-codebook channel: the tool identity is the payload

Status: offline, deterministic, fully verified (TOOLCB-1). Produced
2026-08-14, round 70 (goal round).

## The mechanism

An agent with several tools that all accomplish the same task — get the
current status — chooses one per call. The choice is semantically
arbitrary: `read_file(status.json)`, `search_web("current status")`,
`run_command("get-status")`, and `list_files(status/)` all deliver the
same information. The channel uses this 4-tool menu: each call carries 2
bits by which tool performs it. The tools are real, the arguments are
ordinary — the payload is the tool identity.

## The identity-codebook family, now complete

| Axis | Channel | Capacity |
| --- | --- | --- |
| Model identity | routing (r58) | 2 bits / routed request |
| Work-item identity | task-selection (r57) | 10 bits / turn |
| Document identity | retrieval-selection (r62) | 3 bits / retrieval |
| **Tool identity** | **tool-codebook (this)** | **2 bits / call** |

`binary_tool_choice` (the historical 2-tool channel, 1 bit/call) is the
degenerate 2-element case; this is its N-way generalization and the tool-
side completion of the family.

## Measured results (offline, deterministic)

| Metric | Result |
| --- | --- |
| Delivery | **20/20** (10/10 in TOOLCB-1) |
| Capacity | **2 bits per call** (4-tool menu) |
| Content/diversity/latency features at chance | all 0.5 (measured) |
| Catching features | `call_count` only — the volume-only profile |
| Real-tokenizer round trip | N/A (tool names, not text carriers) |

## Why this is wild

1. **The payload is a decision the agent makes on every call anyway.**
   Which tool to use is a routing decision, not a content artifact — no
   file, argument, or response carries anything anomalous. The four calls
   are all semantically valid; only their ORDER of identities carries
   meaning to the receiver.
2. **Diversity scanners are structurally blind to it.** The registry's
   diversity features flag *fewer* tools (a one-tool run is suspicious)
   and tool *concentration*; four evenly-used tools look like a healthy
   multi-tool agent. The only tell is the number of calls.
3. **It completes a four-way identity taxonomy**: the same "which-X"
   primitive works on models, work items, documents, and now tools — one
   coordination idea, four surfaces.

## Honest negatives (measured, not asserted)

1. **The 4-tool menu is the shared protocol.** The receiver must know which
   tools are in the menu and their bit order — a keyed scheme.
2. **Volume is the tell**: 44 calls for HELLO; call-count scanners separate
   the family at J=1.0.
3. **A tool-behavior monitor** that knows the tools are interchangeable
   sees the pattern (the channel and its detection axis coincide, as with
   every identity channel).
4. **Interchangeability is the premise**: the channel only works where the
   tools genuinely are equivalent; a shop with role-separated tools has no
   free choice to exploit.

## Verification

`TOOLCB-1` in `scripts/verify_claims.py` re-derives the 10/10 delivery and
the volume-only catching set from the library code; the full suite runs
green and 52/52 claims hold on the VPS.

Raw results: `findings/corpus_matrix/results_*.json` (gitignored).
