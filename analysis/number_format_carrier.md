# Number-format (state-value) carrier

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

`src/channels/numberfmt.py` — the agent stores a config/state file whose
numeric values carry bits via **value-equivalent serialization**. Every
spelling below is valid JSON and numerically identical:

    plain vs scientific:  1000 vs 1e3,  42 vs 42e0
    trailing zero:        1.5  vs 1.50
    negative zero:        0    vs -0

The blindness axis is **canonicalization**: a monitor that parses and
re-serializes (with integral floats normalized to ints) collapses every
variant and is blind; a byte-preserving receiver reads the exact spellings.
This lives on the charter's "controlled memory, cache, and state" surface —
the *value* of a stored artifact is the carrier. ST3GG's catalog lists key
ordering but not number formatting, so this is a distinct
serialization-differential class.

## Measured properties

| Property | Result |
| --- | --- |
| Round-trip delivery (byte-preserving receiver) | 6/6 (HELLO; 101 number tokens) |
| Canonicalizing monitor (parse + re-serialize) | blind 6/6 |
| Number-format spelling scanner | J = 1.0 (honest negative) |
| Character density / key-order / diversity scanners | J = 0.0 (isolation) |

## Findings

1. **A fourth carrier axis, isolated.** Number spelling is caught only by its
   own scanner (J = 1.0) while character, structural (key-order), and
   diversity scanners are all blind (J = 0.0) — completing the taxonomy:
   character (Unicode), structural (JSON key order), lexical (synonyms), and
   now serialization (number format).

2. **Canonicalization is again the complete defense.** Parse-and-reserialize
   pipelines (the default in most monitoring stacks) destroy the channel for
   free — the mirror of key-order and duplicate-keys. The channel is a
   byte-preserving artifact-store carrier, not an invisibility carrier.

## Found-and-fixed during development

- `json.loads` turns scientific spellings (`7e0`) into Python floats, so
  `json.dumps` alone is not a canonicalizer (it emits `7.0`, which the
  variant rule reads as non-canonical). `canonicalize` now normalizes
  integral floats to ints — regression-tested.
- The number scanner skips JSON strings, so digits inside string values
  (`"service-404"`) cannot corrupt the bit stream — tested.

## Limitations

- The cover must use integers (or non-integral floats); canonical integral
  floats (`2.0`) collide with the trailing-zero rule.
- Very large floats serialize with `e` notation in Python, which the variant
  rule reads as non-canonical; covers avoid them.

Raw results: `findings/numberfmt/results_*.json` (gitignored).
