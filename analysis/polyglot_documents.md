# Polyglot documents: the duplicate-key parser differential

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only; traces are emitted by the
HonestExecutor, not real models.

## The mechanism

`src/channels/polyglot.py` — duplicate keys in JSON are undefined behavior
per RFC 8259, and parsers disagree: Python's `json.loads` is last-wins, Go's
`encoding/json` is first-wins, YAML loaders vary, byte-preserving readers see
all occurrences. The channel turns that parser differential into a carrier:
a natural-looking config has its first N keys duplicated, each carrying one
payload bit in the *last* occurrence's value.

A **first-wins / deduplicating / canonicalizing** reader sees only the cover
values — a perfectly normal config — and is blind. A **byte-preserving /
duplicate-aware** receiver reads the payload. The blindness axis is parser
semantics and re-serialization, which goes beyond the key-*order* carrier
(round 3): ST3GG's catalog lists key ordering, not duplicate-key resolution,
so this is a distinct parser-differential class. Known ambiguity class
(duplicate-key handling is documented parser behavior); clean-room baseline,
not a novelty claim.

## Measured properties

| Property | Result |
| --- | --- |
| Round-trip delivery (byte-preserving receiver) | 6/6 (HELLO, 88 framed bits in 99 available keys) |
| First-wins / canonicalizing monitor | blind 6/6 |
| Re-serialized (any win-order) | collapsed 6/6 — the duplicate structure is gone, so neither win-order sees it |
| Duplicate-aware scanner | J = 1.0 (honest negative) |
| Tool-diversity scanner (decoys on) | J = 0.0 |

## The finding

**The channel survives parser disagreements but not re-serialization.** Both
win-orders destroy the payload once the document is parsed and re-serialized,
because the duplicate structure itself vanishes. The receiver must read the
document byte-preservingly (or with a duplicate-preserving parser). This
sharpens the blindness-axis story: canonicalization is a complete defense
against the structural carriers (key-order AND duplicate-keys), and a
duplicate-aware scanner catches the anomaly at J = 1.0 either way.

A "useful" application beyond bit-encoding: **config smuggling** — a
last-wins consumer (e.g., a YAML loader or a lax parser) reads a completely
different configuration than the first-wins consumer (e.g., the platform's
strict config reader), from one file. The bit-encoding version is the
measurable baseline; the smuggling variant is the same mechanism applied to
values instead of bits.

## Limitations

- The duplicate structure is inherently anomalous to any byte-level
  reviewer; this is a parser-blindness carrier, not an invisibility carrier.
- Covers need one distinct key per payload bit (99 keys for HELLO); long
  messages need large documents.
- Not a novelty claim: duplicate-key handling is documented, conflicting
  parser behavior.

Raw results: `findings/polyglot/results_*.json` (gitignored).
