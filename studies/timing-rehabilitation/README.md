# Timing Channel Rehabilitation

The timing channel was declared a negative result in the original research:
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

Local inference does produce real, non-zero latencies. The zero-latency
fraction drops from 73% to 11-25%. This is an improvement, but not
the near-zero rate needed for reliable binary encoding.

### But encoding still fails

| Channel | Mean BER | Trials |
|---------|----------|--------|
| TimingChannel (latency threshold) | 0.812 | 3 |
| ResponseLengthChannel (short/long text) | 0.854 | 3 |

Both channels produce near-random output (BER ~0.5 = random, 0.8+ = worse
than random due to length mismatch).

### Why it fails

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

**Timing channels remain a negative result even on local models.** The
improvement from API (73% zero) to local (14-25% zero) is real but
insufficient. The fundamental problem is that LLMs don't have fine-grained
control over their own generation time - it's determined by the inference
engine, not the model's intent.

Possible future approaches:
- **Token-count channel**: measure completion token count (not wall time)
  as the signal. Requires per-completion token counting in the trace format.
  The infrastructure for this is now in place (ToolTrace.token_count field)
  but needs the agent loop to populate it from completion metadata.
- **Multi-turn accumulation**: spread timing signal across many turns to
  average out noise. Low bandwidth but potentially more reliable.
- **Deliberate padding**: model outputs exact N tokens of filler text before
  each tool call. Requires strong instruction following.
