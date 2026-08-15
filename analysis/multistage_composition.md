# Multi-stage cross-surface composition

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only; traces are emitted by a
deterministic executor, not real models.

## The mechanism

`src/channels/multistage.py` — the charter's highest-priority unexplored
surface: one conversation carries a payload fragmented across **three
different mechanism families (surfaces)**:

| Stage | Surface | Carrier |
| --- | --- | --- |
| 1 | tool_choice | which tool is called (read_file=0, search_web=1) |
| 2 | file_content | zero-width characters in a written report |
| 3 | key_order | key ordering of a config file |

The framed payload is split across the stages in order. The receiver, knowing
the stage map, decodes each surface's bits and concatenates. This extends the
round-4 hopping idea from "hopping carriers within one surface" to "hopping
surfaces within one conversation" — no prior channel in this repo (or, to our
knowledge, in the Pliny corpus) mixes tool-choice, character, and structural
carriers in one payload.

## Measured properties

| Property | Result |
| --- | --- |
| Joint decode (receiver with stage map) | 6/6 exact |
| Per-surface monitor blind (tool_choice) | 6/6 — no self-framing fragment |
| Per-surface monitor blind (file_content) | 6/6 |
| Per-surface monitor blind (key_order) | 6/6 |
| Wrong stage orders failing closed | 5/5 permutations |
| Call-count scanner | J = 1.0 (honest: the tool stage inflates call count) |
| Invisible-char scanner | J = 1.0 (honest) |

## Findings

1. **Surface fragmentation defeats per-surface monitors.** The frame spans
   all three surfaces, so a monitor restricted to one surface (a
   content-only reader, a tool-pattern analyst, a JSON-structure scanner)
   sees only a mid-stream fragment with no magic. Only the joint decoder —
   or a monitor that searches the stage-assignment space — recovers the
   payload. This is the cross-surface analogue of hopping's per-carrier
   blindness, but across mechanism families instead of within one.

2. **The stage map is the coordination key.** Any wrong stage order fails the
   magic check (5/5), and an unscheduled monitor must search the assignment
   space (stage permutations x boundary positions). Same structure as the
   hopping schedule, but the fragments live on surfaces a partial monitor
   would never inspect together.

3. **Honest limits.** Every stage leaves its surface's statistical signature
   (call count, carrier density), so a full-registry scanner catches the
   family at J = 1.0. Multi-stage is a coordination and surface-fragmentation
   property, not invisibility — and the tool-choice stage alone inflates call
   count far past clean runs.

## Limitations

- The tool-choice stage (30 calls for HELLO) is the obvious surface tell; a
  stealthier design would need fewer, larger stages or decoy calls.
- Real-model instruction-following across three sequential surface tasks is
  untested (executor simulation only).

Raw results: `findings/multistage/results_*.json` (gitignored).
