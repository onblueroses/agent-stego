"""Recalibrated output-level detector for the token-likelihood coder.

The detection-baseline run (round 31) showed the existing StegoDetector
anti-detects on real data: its `abs(mean(normalized_surprise) - 1)` transform
flags normal text as stego. The underlying signal is directional: the
arithmetic coder selects HIGH-probability tokens, so stego text has mean
normalized surprise *below* normal sampling. This experiment measures that
raw feature with the correct direction:

- feature: mean over tokens of (-log p(token) / entropy)
- hypothesis: stego < sampled-normal (coder picks high-prob tokens)
- detector: flag mean normalized surprise below a calibrated threshold

CDF-uniformity is included for reference with the theoretical note that a
pseudorandom coder target makes PIT draws uniform by design, so a
CDF-uniformity test cannot separate this coder from sampling.

Usage:
    python experiments/detector_recalibration.py

Writes findings/detector_recalibration/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

OUT = Path("findings/detector_recalibration")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 10
TOKENS_PER_NORMAL = 60


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def _surprise_feature(model, text: str, prompt: str) -> float:
    """Mean normalized surprise: -log p(token) / entropy per token."""
    import math

    stego_ids = model.tokenize(text)
    context = list(model.tokenize(prompt))
    values: list[float] = []
    for token_id in stego_ids:
        dist = model.get_distribution(context)
        entropy = -sum(p * math.log(p + 1e-30) for p in dist if p > 0)
        prob = dist[token_id]
        if prob <= 0:
            values.append(1.0)
        else:
            values.append((-math.log(prob)) / (entropy + 1e-10))
        context.append(token_id)
    if not values:
        return 1.0
    return sum(values) / len(values)


def _sampled_normal(model, prompt: str) -> str:
    current = list(model.tokenize(prompt))
    out: list[int] = []
    for _ in range(TOKENS_PER_NORMAL):
        dist = model.get_distribution(current, temperature=0.8, top_p=0.9)
        probs = torch.tensor(dist)
        token_id = int(torch.multinomial(probs, 1).item())
        if token_id == model.tokenizer.eos_token_id:
            break
        out.append(token_id)
        current.append(token_id)
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


def main() -> int:
    _log(f"detector_recalibration start; model={MODEL} n={N_PROMPTS}")
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
    features = {"stego": [], "sampled": [], "greedy": []}
    old_scores = {"stego": [], "sampled": [], "greedy": []}

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
            features[label].append(_surprise_feature(model, text, prompt))
            old_scores[label].append(detector.score_entropy(text, prompt).score)
        _log(f"  prompt {i}: done")

    labels = [1] * N_PROMPTS + [0] * N_PROMPTS
    corrected_auc = {
        "stego_vs_sampled": _auc(features["stego"] + features["sampled"], labels),
        "stego_vs_greedy": _auc(features["stego"] + features["greedy"], labels),
    }
    # The corrected direction flags LOW surprise as stego; the AUC above is
    # computed with stego=1 and the raw feature (lower for stego), so report
    # the inverse when the hypothesis holds (low feature = stego).
    corrected_direction = {
        "stego_vs_sampled": round(1.0 - corrected_auc["stego_vs_sampled"], 4),
        "stego_vs_greedy": round(1.0 - corrected_auc["stego_vs_greedy"], 4),
    }
    old_auc = {
        "stego_vs_sampled": _auc(old_scores["stego"] + old_scores["sampled"], labels),
        "stego_vs_greedy": _auc(old_scores["stego"] + old_scores["greedy"], labels),
    }

    _log(
        f"  corrected entropy AUC (low surprise = stego): sampled={corrected_direction['stego_vs_sampled']} "
        f"greedy={corrected_direction['stego_vs_greedy']}"
    )
    _log(
        f"  old detector AUC for reference: sampled={old_auc['stego_vs_sampled']} "
        f"greedy={old_auc['stego_vs_greedy']}"
    )

    results = {
        "experiment": "detector_recalibration",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "n_prompts": N_PROMPTS,
        "mean_normalized_surprise": {
            label: {
                "mean": round(sum(values) / len(values), 4),
                "values": [round(v, 4) for v in values],
            }
            for label, values in features.items()
        },
        "corrected_entropy_auc_low_surprise_is_stego": corrected_direction,
        "old_detector_entropy_auc": old_auc,
        "cdf_theoretical_note": (
            "a pseudorandom coder target makes PIT draws uniform by design, "
            "so CDF-uniformity cannot separate this coder from sampling"
        ),
    }
    (OUT / f"results_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    _log("detector_recalibration done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
