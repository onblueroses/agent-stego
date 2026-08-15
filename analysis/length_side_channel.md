# Response-length side channel

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

`src/channels/length_side.py` — the historical token-count timing variant
asked the model to control its verbosity; the framework-level reframe makes
the harness control the completion length attributed to each call (10 vs 100
tokens), and bits ride in the per-call token count. Content-independent: the
response *text* is ordinary, the *length* is the carrier.

## Family table (4 members, measured)

| Member | Content-blind | Modality monitor J | Volume tell J |
| --- | --- | --- | --- |
| timing (clock axis) | 9/9 | latency_std 1.0 | 1.0 |
| resource (compute axis) | 9/9 | cpu_ms_std 1.0 | 1.0 |
| **length (usage axis)** | 9/9 | token_count_std 1.0 | 1.0 |
| error (outcome axis) | 9/9 | error_fraction 1.0 | 1.0 |

## Findings

1. **The family is complete at four members.** Every content-independent
   observable in the harness — clock, compute, usage, outcome — carries the
   same channel structure: content guards blind, a modality-aware monitor
   catches it, framing overhead is a universal volume tell.

2. **Stealth ordering is by instrumentation cost.** timing ~ resource need
   modality-specific monitors (latency histograms, CPU attribution);
   **length** uses standard API telemetry (completion tokens are in every
   usage response) so a usage-aware defender catches it almost for free;
   error is loudest (reliability tooling tracks failures already). The
   ordering: timing ~ resource > length > error.

3. **A framing-independent defender baseline exists.** Since every member
   pays the 48-bit framing overhead, a plain call-count scanner catches all
   four at J = 1.0 — the cheapest defense against the whole family.

Raw results: `findings/behavioral_family/results_*.json` (gitignored).
