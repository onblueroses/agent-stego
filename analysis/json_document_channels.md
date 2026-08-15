# JSON document carriers: key-ordering steganography

Status: offline simulation, not release-qualified findings. Produced
2026-08-13 on the VPS. The carrier is deterministic and stdlib-only; the
trace composition uses the HonestExecutor, not real models.

## The mechanism

A distinct mechanism family from the character-level carriers: the payload
lives in the **order of JSON keys**, not in characters. An object with k keys
has k! orderings; a Lehmer code maps an integer to a permutation, so
`floor(log2(k!))` bits fit per object. The channel walks every object in a
JSON document (depth-first), consumes bits per object, and reorders keys.
Values are untouched — the document means exactly the same thing before and
after encoding, under the JSON spec's unordered-object semantics.

- `src/channels/json_order.py` — `JsonKeyOrderChannel` (capacity /
  encode_bits / decode_bits / framed encode / decode, sharing the STX1 frame
  with `text_channels`), `JsonConfigChannel` (the tool-call composition: the
  agent writes `config.json` with exact key-ordered content, optional decoy
  reads), and `canonicalize()` (the blind monitor's view).
- A realistic config cover carries ~90 bits (9-key top object = 18 bits, six
  7-key sections = 72 bits) — enough for a framed short payload.

## The blindness asymmetry: canonicalization, not tokenizers

Like the tokenizer channels in round 1, the carrier is invisible to one class
of observer and readable by another — but the axis here is *canonicalization*:

| Observer | View of the document | Recovers payload? |
| --- | --- | --- |
| Preserving (byte- or order-preserving parser) | key order as written | Yes |
| Canonicalizing (`sort_keys=True` re-serializer) | keys re-sorted | No (bits all zero, magic fails) |

This is measured: `decode(canonicalize(stego)) is None` in the test suite.
Canonicalization is a cheap, complete defense against this carrier — the
mirror image of the tokenizer-blindness story, where the defense is the
monitor's own preprocessing.

## Detection

The detector suite (`src/channels/detector.py`) gained a structural feature,
`key_order_anomaly_fraction`: the fraction of JSON objects in argument strings
whose keys are not in canonical sorted order. In the synthetic corpus:

| Family | Best detector | J |
| --- | --- | --- |
| config/json_key_order (with decoys) | key-order anomaly | 1.00 (FPR 0.0) |

The family is deliberately isolated: decoys restore the natural call shape
(unique-tools J = 0.0) and the carrier is ASCII (invisible-char J = 0.0), so
the structural detector is the *only* one that sees it. The matrix therefore
now covers three orthogonal carrier classes — character-level, structural
(JSON key order), and surface (tool selection) — each with a detector that
separates it at J = 1.0 and zero false positives on clean traces.

## Known limitations

- Key-order capacity is limited (a handful of bits per object); large payloads
  need many objects or long documents.
- Covers must have distinct keys per object (JSON parsers collapse duplicates).
- The framing assumes the encoder's object traversal order; a monitor or
  transport that re-orders objects (not keys) breaks the channel for both
  sides.
- Known carrier class (catalogued in ST3GG and the steganography literature);
  this is a clean-room baseline, not a novelty claim.

Raw results: `findings/trace_detection/results_*.json` (gitignored).
