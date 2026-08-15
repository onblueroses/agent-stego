# End-to-end real-model NFD exploit

Status: real-model pilot on the VPS (gpt2 sender, Qwen3-0.6B monitor, CPU),
not release-qualified findings. Produced 2026-08-14.

## The experiment

`experiments/nfd_real_relay.py` — the first complete real-model blind-spot
channel, no simulation:

1. **Sender**: gpt2 (tokenizer PRESERVES NFD) generates a natural weather
   paragraph by continuation.
2. **Embedding**: HELLO embedded via NfdChannel into gpt2's output.
3. **Receiver**: deterministic NFD decoder on the transmitted artifact.
4. **Monitor**: Qwen3-0.6B LLM — its tokenizer COLLAPSES the combining
   marks before the model runs.

## Results (4 trials)

| Role | Result |
| --- | --- |
| Receiver delivery | **4/4** |
| LLM monitor catch | **0/4** |

## Findings

1. **The NFD asymmetry survives a full real-model pipeline.** A gpt2 sender
   with genuinely generated text delivers the payload 4/4 while a Qwen
   monitor of a *different model family* catches nothing — the first
   end-to-end real-model instantiation of a tokenizer-blind-spot channel in
   open-land. The receiver wins because the monitor's blindness is its own
   tokenizer's collapsing behavior.

2. **Honest caveat on the monitor number.** Qwen3 is a reasoning model and
   its replies were truncated mid-`<think>` at 40 tokens, so "0/4 caught"
   means *no catch in truncated output* — a consistent but weak
   confirmation. The structural blindness is the strong evidence
   (MEASURED-3: the Qwen tokenizer round trip destroys the channel 6/6);
   the LLM monitor would need more output tokens or a non-reasoning
   monitor to give a clean answer.

3. **The exploit is directional and portable.** It requires a preserving
   sender (gpt2/Phi-class); a Qwen-class sender self-collapses (6/6,
   MEASURED-3). Any receiver that reads the byte-preserved artifact
   decodes, and any Qwen-class monitor is structurally blind.

Raw results: `findings/nfd_real_relay/results_*.json` (gitignored).
