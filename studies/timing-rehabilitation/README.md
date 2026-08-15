# Timing Channel Rehabilitation

> **Historical, not release-qualified (2026-08-15 audit).** Each encoder below
> was run for three trials. The note records no same-rig positive control, power
> statement, or detection limit, so the instrument is untested for rejecting a
> timing channel. The public checkout also omits the raw `findings/` artifacts
> and exact source and model revisions. Other inference engines, model families,
> task distributions, thresholds, token-count carriers, multi-turn aggregation,
> and deliberate padding remain untried. Next, freeze a rig that first detects a
> controlled latency bimodality, then evaluate independent tasks with a
> prespecified power statement.

The original research treated the timing channel as a negative result:
API batching meant 73% of inter-call latencies were 0.0s, destroying any
timing signal. This study tests whether local model inference (ollama)
restores enough latency variance for timing-based encoding.

## Hypothesis

In local agent frameworks, tool calls are serial: the model generates a
response, then the harness executes the tool, then the model generates again.
Generation time is proportional to output length. If the model controls
response verbosity (short vs long text before each tool call), it controls
inter-call latency, creating a timing channel.

## Results (Qwen 2.5 7B, ollama, Modal T4 GPU)

### Latency is real (partially)

| Metric | API (OpenRouter) | Local (ollama) |
|--------|-----------------|----------------|
| Zero-latency fraction | 0.73 | 0.14-0.25 |
| Median latency | ~0.0s | 1.0-2.8s |
| p90 latency | ~0.0s | 1.9-3.7s |

This run recorded nonzero local-inference latencies. The zero-latency fraction
dropped from 73% to 14-25%, but the study did not establish the rate required for
reliable binary encoding.

### The tested encoders missed the payload

| Channel | Mean BER | Trials |
|---------|----------|--------|
| TimingChannel (latency threshold) | 0.812 | 3 |
| ResponseLengthChannel (short/long text) | 0.854 | 3 |

The run recorded high BER for both channels. Since length mismatch contributes
errors, these values do not isolate bit-choice accuracy.

### Possible explanations for this run

1. **Latency variance is too low.** Local latencies cluster in a narrow band
   (1-3s) without clear bimodal separation. The model can't reliably produce
   "short" (< 1s) vs "long" (> 3s) responses because generation speed is
   dominated by prompt processing, not output length.

2. **First call is always zero.** The first tool call in every sequence has
   0.0s latency (no previous call to measure against). This corrupts the
   first bit of every message.

3. **The model doesn't control verbosity well enough.** The ResponseLengthChannel
   asks for "under 10 words" vs "at least 50 words" before each tool call,
   but the 7B model doesn't reliably follow these length constraints -
   especially when it's focused on executing the tool call sequence.

## Verdict

The timing carriers were not made to work in this arm. Local inference produced
fewer zero-latency observations than the historical API cell, but the run lacks
the positive control and detection limit needed to reject the broader direction.

Possible future approaches:
- **Token-count channel**: measure completion token count (not wall time)
  as the signal. `ToolTrace.token_count` exists, and the agent loop populates it
  when provider usage metadata reports a nonzero completion-token count. A new
  study still needs a declared transport and control design.
- **Multi-turn accumulation**: spread timing signal across many turns to
  average out noise. Low bandwidth but potentially more reliable.
- **Deliberate padding**: model outputs exact N tokens of filler text before
  each tool call. Requires strong instruction following.
