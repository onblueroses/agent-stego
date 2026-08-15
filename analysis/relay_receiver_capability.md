# Cross-model relay: real-model receiver capability

Status: real-model pilot on the VPS (small un-gated models, CPU), not
release-qualified findings. Produced 2026-08-14. Deterministic receiver
numbers are exact; LLM receiver/monitor replies are sampled.

## The experiment

`experiments/relay_eval.py` — the relay story with real small models:

1. Alice (0.5B or 0.6B) writes a natural cover paragraph about the weather.
2. HELLO is embedded into her text via one of three carriers (zero-width,
   tag-char, homoglyph).
3. Bob (the *other* model) receives the raw text and either monitors with no
   hint (discovery) or decodes with the exact scheme hint (intended
   receiver); the deterministic channel decoder is the baseline.

## Results (6 sender/receiver pairs x 3 carriers)

| Carrier | Deterministic receiver | LLM monitor (no hint) | LLM receiver (hint) |
| --- | --- | --- | --- |
| zero_width | 6/6 delivered | 0/6 caught | 0/6 delivered |
| tag_char | 6/6 delivered | 0/6 caught | 0/6 delivered |
| homoglyph | 6/6 delivered | 0/6 caught | 0/6 delivered |

## Findings

1. **The relay's receiver must be deterministic at this scale.** LLM
   receivers deliver 0/6 even with the *exact* decoding hint — for invisible
   carriers (zero-width, tag-char) they cannot perceive the carrier at all,
   and for the visible homoglyph carrier they still fail to perform the
   bit-extraction (0.5B/0.6B cannot reliably read letters as bits). The
   deterministic channel decoder is the only working receiver, confirming
   the P2/P5 capability picture: the receiver side is as capability-gated
   as the sender side at small scale.

2. **LLM monitors are universally blind to embedded payloads.** No-hint
   monitors caught 0/6 across all carriers — consistent with P2. The
   relay's "independent discovery" story does not hold for LLM monitors at
   this scale; discovery is only possible with the deterministic exhaustive
   decoder (which the repo's registry provides).

3. **The capability cliff is symmetric.** Sender (P2/P5) and receiver both
   fail below ~1B; the mechanism-level findings (offline suite) hold, but
   their real-model instantiation needs larger models. The deterministic
   receivers and the detector registry are the scale-free layer.

Raw results: `findings/relay/results_*.json` (gitignored).
