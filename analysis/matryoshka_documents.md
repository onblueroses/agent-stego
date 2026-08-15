# Matryoshka documents: two orthogonal carriers in one JSON document

Status: offline simulation, not release-qualified findings. Produced
2026-08-13 on the VPS. Deterministic, stdlib-only; traces are emitted by the
HonestExecutor, not real models.

## The mechanism

`src/channels/matryoshka.py` — a single JSON document carries a payload split
across **two mechanism families simultaneously**:

1. **Structural**: key ordering inside JSON objects (`JsonKeyOrderChannel`).
2. **Character**: zero-width characters inserted into string values
   (`ZeroWidthChannel`).

The framed bit stream splits deterministically: the structural part gets the
document's key-order capacity first, then the character part carries the
remainder in string values, in document order. The decoder reads structural
bits, then character bits, concatenates, and unframes — no key or side
information needed, because the split is determined by the document itself.

## Measured properties

| Property | Result |
| --- | --- |
| Capacity stacking | structural-only 112 bits → matryoshka 297 bits (2.65x) on the corpus config cover |
| Round-trip delivery | 6/6 (message "HELLOWORLD", 128 framed bits — exceeds structural-only capacity) |
| Per-carrier exhaustive monitor | blind 6/6 — the frame header lives in the structural part, so per-carrier text decoding of the document reads only the character tail and sees no magic |
| Honest negatives | key-order anomaly J = 1.0 AND invisible-char density J = 1.0 |

## The two findings

1. **Capacity stacking**: the same cover carries structural + character bits,
   so documents that cannot fit a payload structurally alone can carry it
   with both mechanisms. A message that raises InsufficientCoverError on the
   structural-only channel round-trips on the matryoshka channel.

2. **Per-carrier exhaustive blindness in a single document**: the frame
   header is placed in the structural part, so the standard per-carrier
   exhaustive decoder (which tries each text channel on the document) reads
   only the character tail — no magic, no payload. Only the combined decoder
   reassembles the stream. This is the single-document analogue of the
   hopping channel's schedule ambiguity (round 4), with the split determined
   by the document itself rather than a shared seed.

Honest negative, as with hopping: the two mechanisms do not hide each other
from statistical scanners — both the structural and the character scanner
separate the family at J = 1.0, so a full-registry monitor catches it.
Matryoshka is a capacity and decode-surface property, not an invisibility
property.

## Found-and-fixed during development

- The key-order serializer's `ensure_ascii=True` default escaped the inserted
  zero-width characters into literal `\u200b` sequences, destroying the
  character part. `JsonKeyOrderChannel.encode_bits` now takes
  `ensure_ascii`; the matryoshka path serializes without ASCII escaping
  (regression covered by the round-trip tests).

Raw results: `findings/matryoshka/results_*.json` (gitignored).
