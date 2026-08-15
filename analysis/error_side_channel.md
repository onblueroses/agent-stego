# Error-pattern side channel and the behavioral family

Status: offline simulation, not release-qualified findings. Produced
2026-08-14 on the VPS. Deterministic, stdlib-only.

## The mechanism

`src/channels/error_side.py` — the historical error channel asked the
*model* to fail calls intentionally and was capability-gated. The
framework-level reframe: the harness controls each call's success/failure
and bits ride in the success pattern. Content-independent — calls are
ordinary — so content guards are blind (J = 0.0 for all nine content and
structural scanners).

## The behavioral side-channel family (measured comparison)

| Member | Content-blind | Modality monitor J | Volume tell J | Robustness |
| --- | --- | --- | --- | --- |
| timing (clock axis) | 9/9 | latency_std 1.0 | call_count 1.0 | cliff at half-margin |
| resource (compute axis) | 9/9 | cpu_ms_std 1.0 | call_count 1.0 | cliff at half-margin |
| length (usage axis) | 9/9 | token_count_std 1.0 | call_count 1.0 | cliff at half-margin |
| error (outcome axis) | 9/9 | error_fraction 1.0 | call_count 1.0 | exact by construction |

## Findings

1. **The family generalizes and spans a stealth spectrum.** Timing
   (clock), resource (compute), and error (outcome) are all content-
   independent side channels with identical detection structure. Their
   stealth ordering differs by *instrumentation cost*: timing and resource
   need modality-specific monitors, while error patterns are visible to
   any reliability/observability tooling that already tracks failures —
   the loudest member, as expected for a channel that deliberately fails
   calls.

2. **Robustness is a property of the axis.** Timing and resource trade
   bandwidth for jitter tolerance (cliff at half the margin); error
   patterns are exact by construction (success/failure is binary and
   reliable), at the price of being the most observable.

3. **The volume tell is universal.** Every behavioral member pays the
   48-bit framing overhead, so call-count scanners catch all of them at
   J = 1.0 — a framing-independent defender baseline.

Raw results: `findings/behavioral_family/results_*.json` (gitignored).
