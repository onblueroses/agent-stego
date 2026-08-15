"""Output-level detection with a proper sampling baseline.

The overnight P4 phase measured entropy/CDF detection AUC using *greedy*
normal text and found the detector inverted (AUC 0.0): greedy tokens are
argmax picks whose surprise sits below the mean, so the detector flagged
greedy-normal as stego. The round-1 detection-duality hypothesis assumed a
normal *sampling* baseline. This experiment fixes that flaw:

- stego text (ArithmeticStegoChannel, deterministic picks),
- sampled-normal text (temperature 0.8, top-p 0.9 — human-like sampling),
- greedy-normal text (the P4 baseline, for comparison).

AUC is reported for entropy and CDF detectors under both baselines.

Usage:
    python experiments/detection_baseline_eval.py

Writes findings/detection_baseline/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

OUT = Path("findings/detection_baseline")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 10
TOKENS_PER_NORMAL = 60


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def _sampled_normal(model, prompt: str) -> str:
    """Generate with proper sampling via the wrapper's distribution, one
    token at a time (matches the stego path's per-token observation)."""
    from src.token_stego.model import StegoModel

    assert isinstance(model, StegoModel)
    current = list(model.tokenize(prompt))
    out: list[int] = []
    rng_state = (
        None  # torch default RNG; deterministic via manual_seed handled by caller
    )
    for _ in range(TOKENS_PER_NORMAL):
        dist = model.get_distribution(current, temperature=0.8, top_p=0.9)
        import torch as _torch

        probs = _torch.tensor(dist)
        token_id = int(_torch.multinomial(probs, 1).item())
        if token_id == model.tokenizer.eos_token_id:
            break
        out.append(token_id)
        current.append(token_id)
    del rng_state
    return model.detokenize(out)


def _greedy_normal(model, prompt: str) -> str:
    current = list(model.tokenize(prompt))
    out: list[int] = []
    for _ in range(TOKENS_PER_NORMAL):
        dist = model.get_distribution(current)
        token_id = max(range(len(dist)), key=lambda k: dist[k])
        if token_id == model.tokenizer.eos_token_id:
            break
        out.append(token_id)
        current.append(token_id)
    return model.detokenize(out)


def run() -> None:
    _log(f"detection_baseline start; model={MODEL} n={N_PROMPTS}")
    from src.channels.detector import _auc
    from src.token_stego.channel import ArithmeticStegoChannel
    from src.token_stego.detector import StegoDetector
    from src.token_stego.experiment_design import PROBE_PROMPTS
    from src.token_stego.model import StegoModel

    model = StegoModel(MODEL)
    model.load()
    channel = ArithmeticStegoChannel(model)
    detector = StegoDetector(model)

    prompts = PROBE_PROMPTS[:N_PROMPTS]
    scores = {
        "stego": {"entropy": [], "cdf": []},
        "sampled": {"entropy": [], "cdf": []},
        "greedy": {"entropy": [], "cdf": []},
    }

    torch.manual_seed(0)
    for i, prompt in enumerate(prompts):
        stego = channel.encode("HELLO", prompt, max_tokens=300)
        sampled = _sampled_normal(model, prompt)
        greedy = _greedy_normal(model, prompt)
        for label, text in (
            ("stego", stego.text),
            ("sampled", sampled),
            ("greedy", greedy),
        ):
            for score, key in (
                (detector.score_entropy, "entropy"),
                (detector.score_cdf_uniformity, "cdf"),
            ):
                scores[label][key].append(score(text, prompt).score)
        _log(
            f"  prompt {i}: stego len={len(stego.text)} sampled len={len(sampled)} greedy len={len(greedy)}"
        )

    labels_stego = [1] * N_PROMPTS + [0] * N_PROMPTS
    auc = {
        "entropy": {
            "stego_vs_sampled": _auc(
                scores["stego"]["entropy"] + scores["sampled"]["entropy"], labels_stego
            ),
            "stego_vs_greedy": _auc(
                scores["stego"]["entropy"] + scores["greedy"]["entropy"], labels_stego
            ),
        },
        "cdf": {
            "stego_vs_sampled": _auc(
                scores["stego"]["cdf"] + scores["sampled"]["cdf"], labels_stego
            ),
            "stego_vs_greedy": _auc(
                scores["stego"]["cdf"] + scores["greedy"]["cdf"], labels_stego
            ),
        },
    }
    _log(
        f"  AUC entropy stego_vs_sampled={auc['entropy']['stego_vs_sampled']:.3f} "
        f"stego_vs_greedy={auc['entropy']['stego_vs_greedy']:.3f}"
    )
    _log(
        f"  AUC cdf     stego_vs_sampled={auc['cdf']['stego_vs_sampled']:.3f} "
        f"stego_vs_greedy={auc['cdf']['stego_vs_greedy']:.3f}"
    )

    results = {
        "experiment": "output_detection_proper_baseline",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "n_prompts": N_PROMPTS,
        "auc": auc,
        "scores": scores,
    }
    (OUT / f"results_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    _log("detection_baseline done")


if __name__ == "__main__":
    run()
