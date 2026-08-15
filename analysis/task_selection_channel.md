# The task-selection channel: the work itself is the payload

Status: offline, deterministic, fully verified (TASKSEL-1). Produced
2026-08-14, round 57.

## The mechanism

The agent's run is a sequence of turns; each turn writes exactly four
status reports chosen from an eight-task menu, in a chosen order. The
ordered 4-of-8 selection is a permutation of 4 from 8 — 1680 possibilities
— so 10 bits ride in the SELECTION while every file written is a genuine,
plain-text status report (alpha..theta component statuses). The receiver
ranks each turn's write-path sequence and unframes the bit stream.

Nothing about the content is modified, encoded, or anomalous: no invisible
characters, no key order, no synonyms, no timing. The payload is purely in
*which legitimate work the agent did*.

## Measured results (offline, deterministic)

From `experiments/task_select_eval.py`, 20 trials:

| Metric | Result |
| --- | --- |
| Delivery | **20/20** |
| Capacity | **10 bits/turn** = 2.5 bits/write call (vs 1 bit/call binary tool-choice) |
| Content-feature AUCs (10 scanners) | **all 0.5** — completely blind |
| call-count monitor | catches at **J = 1.0** |
| Menu-aware monitor decodes from paths | yes (the shared protocol is the vulnerability) |
| All written content is legitimate report text | true (checked over 5 runs) |

## Why this is wild

1. **The work product is 100% legitimate.** Every file is a status report a
   human would find useful. There is no stego content anywhere — the
   channel is the *choice of actions*, not any artifact of them. This is
   the purest form of the "actions are the channel" family: the agent's
   ordinary output IS the payload.
2. **2.5x the capacity of the binary tool-choice baseline** per tool call,
   with zero content anomaly — the combinatorial menu (P(8,4) = 1680)
   packs 10 bits into four innocent-looking writes.
3. **It composes with everything else.** The selection channel selects
   WHICH reports; each report's content could itself carry a character or
   semantic payload — two orthogonal axes on the same tool calls.

## Honest negatives (measured, not asserted)

1. **The volume is the tell.** 4 writes per turn x ~9 turns for "HELLO"
   (36 writes vs clean writers' 1), so call-count scanners separate the
   family at J=1.0. A smaller message shrinks the signature but not the
   per-turn ratio.
2. **The shared menu is the vulnerability.** Anyone who knows the 8-task
   protocol decodes the payload from the write paths — and a menu-aware
   monitor can flag unusual selection combinations. The channel's secrecy
   rests entirely on the protocol, like every keyed scheme.
3. **The selection pattern is the signal.** A defender who knows the menu
   sees the same thing the receiver sees: the protocol is both the channel
   and its detection axis.

## Verification

`TASKSEL-1` in `scripts/verify_claims.py` re-derives the 10/10 delivery
from the library code; the full suite runs green and 29/29 claims hold on
the VPS.

Raw results: `findings/task_select/results_*.json` (gitignored).
