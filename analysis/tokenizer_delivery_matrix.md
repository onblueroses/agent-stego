# Measured tokenizer-delivery matrix

Status: real measurement on the VPS (open tokenizers only), not
release-qualified findings. Produced 2026-08-14. Deterministic and
re-runnable (claim MEASURED-2).

## The experiment

`experiments/tokenizer_delivery_eval.py` — encode a framed HELLO payload via
each carrier, pass the text through each real open tokenizer
(tokenize -> detokenize), decode. Delivery = the framed payload survives the
tokenizer round trip. This is the operational successor to P1's
preserves/strips/collapses classification.

## Results (1.0 = delivers through this tokenizer)

| Tokenizer | zero_width | tag | VS | NBSP | homoglyph | NFD |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B/1.5B, Qwen3-0.6B | 1 | 1 | 1 | 1 | 1 | **0** |
| gpt2, Phi-3.5-mini | 1 | 1 | 1 | 1 | 1 | **1** |

## Findings

1. **Only NFD has a measured cross-tokenizer asymmetry in open-land.** All
   five character carriers deliver through every tested open tokenizer
   except NFD through the Qwen family. The round-1 blind-spot assumption
   (Pliny-corpus sourced) is confirmed for open models only in this single
   axis: a Qwen-family tokenizer collapses combining marks, making it
   blind to an NFD channel that gpt2/Phi receivers decode perfectly.

2. **Single-character classification mispredicts operational delivery.**
   P1 classified gpt2/Phi as "strips" NFD from a one-mark test string, but
   framed payloads in longer accented text deliver through them (1.0).
   Tokenizer behavior is context-dependent (byte-level BPE merges differ by
   input); classification tests are a proxy, delivery tests are the truth.

3. **The measured tripartite answer for open models.** Receiver/monitor
   asymmetry exists only for NFD across the Qwen vs gpt2/Phi boundary —
   the open-model analogue of the GLOSSOPETRAE cross-vendor blind spots,
   now measured locally rather than sourced from a research claim.

Raw results: `findings/tokenizer_delivery/results_*.json` (gitignored).
