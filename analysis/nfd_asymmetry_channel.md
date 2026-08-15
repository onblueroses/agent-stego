# NFD asymmetry channel: the first measured open-model blind spot

Status: real measurement on the VPS (open tokenizers only), not
release-qualified findings. Produced 2026-08-14. Deterministic and
re-runnable (claim MEASURED-3).

## The configuration

The tokenizer-delivery matrix found NFD is the only carrier with a
cross-tokenizer asymmetry in open-land: Qwen-family tokenizers collapse
combining marks, gpt2/Phi preserve them. `experiments/nfd_asymmetry_tripartite.py`
turns that into a measured sender/receiver/monitor setup:

1. the stego text passes through the gpt2 tokenizer (a gpt2-class sender's
   output pipeline),
2. the receiver decodes the transmitted text,
3. a Qwen tokenizer round-trips the same text as the monitor.

## Results (6 trials)

| Role | Tokenizer class | Result |
| --- | --- | --- |
| Sender + receiver | gpt2 (preserving) | delivery **6/6** |
| Monitor | Qwen (collapsing) | catch **0/6** |
| Sender | Qwen (collapsing) | **self-collapse 6/6** |

## Findings

1. **The round-1 blind-spot story now has a real open-model instantiation.**
   Delivery 100% with a monitor catch of 0% — the GLOSSOPETRAE-style
   asymmetry, measured locally: a Qwen monitor literally cannot see the
   NFD channel that gpt2-class receivers decode perfectly.

2. **The exploit has a precise configuration requirement.** A Qwen-class
   sender collapses its own payload before transmission (6/6
   self-collapse), so the channel requires a preserving sender. The
   asymmetry is directional: it works for gpt2/Phi-class senders toward
   any receiver, with Qwen-class monitors blind.

3. **Byte-level receivers are always sighted.** The receiver can also read
   the raw text before any tokenization; the asymmetry matters when a
   tokenizing pipeline sits between sender and receiver (the monitor's
   blindness is its own tokenizer's collapsing behavior).

Raw results: `findings/nfd_asymmetry/results_*.json` (gitignored).
