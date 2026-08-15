# Self-describing tool traces: in-trace channel headers

Status: offline simulation, not release-qualified findings. Produced
2026-08-13 on the VPS. Deterministic, stdlib-only; traces are emitted by the
HonestExecutor, not real models.

## The pattern

ST3GG's header pattern (magic bytes + serialized config embedded in the
carrier) applied to the tool-call surface. A reserved `check_status` call
carries the contract in its target argument:

    stx1|<channel_key>|<num_chars>

`SelfDescribingChannel` (src/channels/self_describing.py) wraps any
registered channel: encode() prepends the header call to the inner prompt;
decode() parses the header, dispatches to the registered channel in O(1), and
fails closed on key or length mismatch. Seven channels are registered: the
historical table channel (all three modes), permutation, argument-carrier,
file-content (zero-width), and JSON key-order.

## Measured properties

| Channel | Header dispatch | detect_trace_channel | Headerless candidates |
| --- | --- | --- | --- |
| table/path | HI ✓ | (table/path, HI) | 1 |
| permutation | HI ✓ | (permutation, HI) | 3 |
| argument_carrier/homoglyph | HI ✓ | (argument_carrier/homoglyph, HI) | 3 |
| file_content/zero_width | HELLO ✓ | (file_content/zero_width, HELLO) | 1 |
| config/json_key_order | HELLO ✓ | (config/json_key_order, HELLO) | 1 |

- With a header, decoding is O(1) dispatch with no side information.
- Without a header, the analyst must try every registered channel
  (`decode_trace_exhaustive`); overlapping traces (read_file-heavy families)
  produce up to 3 plausible decodes, so the header also resolves ambiguity.
- Fail-closed contract: wrong channel key or wrong declared length in the
  header yields an empty decode, never a wrong message.

## The honest trade-off

The header trades stealth for decodability:

- It adds one reserved call per trace (call-count +1, tool diversity +1).
  The tool-diversity scan does not flag header traces vs the clean corpus
  (J = 0.0 — a `check_status` call keeps unique-tool counts in the natural
  range), but any monitor that reads argument values sees the marker target
  (`stx1|...`) directly.
- Use self-describing headers for interoperability, artifact self-containment,
  and blind-analysis dispatch — not for covert delivery.

## Limitations

- Registered channels are constructed fresh from the header key; custom
  per-instance configuration (covers, file lists) remains side information.
- Real-model prompt compliance with the "call this protocol check first"
  instruction is untested (the strict historical prompts forbid check_status);
  executors follow the wrapper prompt exactly.
- The header does not authenticate the sender; a forger can emit any header.

Raw results: `findings/self_describing/results_*.json` (gitignored).
