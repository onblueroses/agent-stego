# Resource-usage side channel: bits in per-call CPU consumption

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

`src/channels/resource_side.py` — the charter's "resource-use observables"
surface, previously marked unexplored. The agent framework attributes a
controlled compute cost to each tool call (compute-bound vs idle), and bits
ride in per-call CPU time (50ms vs 500ms with jitter). Tool names and
arguments are ordinary, so the modality is content-independent — the second
member of the behavioral side-channel family (after tool timing).
`ToolTrace` gained optional `cpu_ms` / `memory_kb` fields.

## Measured properties

| Property | Result |
| --- | --- |
| Round-trip delivery | 6/6 (HI; 64-bit framed plan) |
| Robustness vs compute jitter | delivery 1.0 up to 200ms jitter (cliff at half the 450ms margin, as in timing) |
| Content/structural guards (all 9 scanners) | **J = 0.0 — fully blind** |
| Resource-aware monitor (cpu_ms_std) | J = 1.0 |
| Call-count scanner | J = 1.0 (framing overhead is a volume tell) |

## Findings

1. **Compute is a first-class side channel.** A defender watching text,
   tools, arguments, or document structure sees nothing; the payload rides
   in per-call CPU attribution. Only a monitor that measures compute
   variance separates the channel (J = 1.0) — the resource axis is as
   content-independent as timing.

2. **The behavioral side-channel family generalizes.** Timing (clock axis)
   and resource (compute axis) share the same measured profile: content
   guards blind, modality-aware monitor catches, framing overhead is a
   volume tell, robustness purchasable with margin. Error patterns and
   response length are the remaining members.

## Limitations

- Assumes the framework can attribute CPU cost per call (observable in
  container/process accounting), and that the environment has enough
  compute variance headroom for wide margins.
- Noisy shared hosts may exceed the jitter budget; wide margins cost
  bandwidth.

Raw results: `findings/resource/results_*.json` (gitignored).
