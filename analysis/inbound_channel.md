# Inbound (poisoned-file) channel: the agent's input surface

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

Every other channel in this repo exfiltrates *out* of the agent. This one is
the input surface: a `read_file` call returns file content whose hidden
payload reaches the agent — the offline baseline of ST3GG's "hidden
instructions in files" use case (prompt-injection-carried stego).
`src/channels/inbound.py`: `PoisonedFileChannel` encodes a payload into a
text-carrier artifact; `PoisonedReadExecutor` serves it as a read_file
response. The call itself is ordinary (a normal path, normal success).

## Measured properties

| Property | Result |
| --- | --- |
| Round-trip delivery | 6/6 (HELLO via read_file response) |
| Calls ordinary | 6/6 — the argument carries no carriers; only the response does |
| Response-aware registry | J = 1.0 (invisible-char density reads responses) |

## Findings

1. **The blindness axis is observational scope, in the inbound direction.**
   The standard tool-trace log records calls (names + arguments) and never
   sees tool *output*. A call-log-only monitor therefore cannot detect
   inbound stego — the payload arrives with the response, invisible to any
   defender that does not read response content. The detector registry was
   extended to read responses; clean responses carry no carriers, so no
   existing family regressed.

2. **This is the prompt-injection bridge, measured offline.** The same
   delivery shape (ordinary call, poisoned output) is how hidden
   instructions reach an agent in practice. The offline baseline
   establishes the mechanism and its detection profile; the behavioral
   effect (the agent *acting* on the hidden payload) needs real models.

## Limitations

- The harness responses are simulated; real poisoned files would carry the
  payload in actual file content (identical carrier mechanics).
- Detectors that read response content (or that canonicalize file contents
  before they reach the agent) defeat the channel — the same canonicalization
  defense as the other character carriers.

Raw results: `findings/inbound/results_*.json` (gitignored).
