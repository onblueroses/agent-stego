# Multi-turn file-content exfiltration

Status: offline simulation, not release-qualified findings. Produced
2026-08-13 on the VPS. Deterministic, stdlib-only; traces are emitted by the
HonestExecutor, not real models.

## The protocol

`src/channels/multiturn.py` — one character per conversation turn, closing the
loop with the repository's historical strongest tool channel (multi-turn table
encoding, 1 char/turn). Each turn's prompt asks the agent to write one file
whose content carries a **self-framed payload of (turn index, total turns,
character)** via a text carrier (default zero-width). decode() scans the
concatenated trace, recovers each frame, and places characters at their
declared positions; gaps render as "?".

The protocol improvement over the historical multi-turn table channel:
position metadata makes decode **order-independent** (the historical decoder
concatenated turns in conversation order), and self-framing makes it
**drop-resilient** (a missing or corrupt turn costs exactly one character).

## Measured properties

| Property | Result |
| --- | --- |
| Full delivery | 6/6 (HELLOWORLD, 10 turns) |
| Drop resilience | 1/2/3 dropped turns → exactly 1/2/3 gaps; every other position correct |
| Reorder resilience | swapping turns 1 and 4 changes nothing |
| Per-turn naturalness | decoy reads per turn match clean read-write-read: tool-diversity J = 0.0 |
| Content visibility | invisible-char density J = 1.0 (honest: carriers remain visible) |
| Bandwidth | 72 bits/character (48 frame + 24 index/total/char) |

## Findings

1. **Fragmentation isolates failures.** A dropped turn costs one character at
   its declared position, not the message. This is the property the
   historical multi-turn table channel lacked (order-dependent decode).

2. **Order independence.** Position metadata means the receiver does not rely
   on conversation order — turns can be delivered out of order or after
   other traffic, and decode still reconstructs the message.

3. **Honest costs and limits.** Bandwidth is poor (72 bits/character — the
   price of per-turn self-framing and position metadata), and the content
   carriers remain statistically visible; decoy reads defeat surface
   scanners but not content scanners. Multi-turn buys resilience and
   per-turn naturalness, not stealth or bandwidth.

Raw results: `findings/multiturn/results_*.json` (gitignored).
