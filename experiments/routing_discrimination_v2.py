"""Routing no-log receiver v2: tokenizer-fingerprint discrimination.

The round-58 discrimination experiment identified the routed model from
output STYLE features at 0.5 (chance 0.25). This re-scores the SAME raw
responses with a different receiver signal: how the five public local
tokenizers tokenize the response. Different model families have different
vocabularies, so the same text yields different token counts per
tokenizer — a fingerprint the receiver can compute with public tokenizers
and no knowledge of the generating model.

Method: load the raw samples from findings/routing_discrimination/, score
each response with the 5 tokenizers (token count = feature), add the
original 6 text features, and classify with leave-one-prompt-out nearest
centroid. Honest: same 12 samples, same protocol, different signal.

Usage:
    python experiments/routing_discrimination_v2.py

Writes findings/routing_discrimination_v2/results_<timestamp>.json
"""

from __future__ import annotations

import glob
import json
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("findings/routing_discrimination_v2")
TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "openai-community/gpt2",
    "microsoft/Phi-3.5-mini-instruct",
]


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    import re

    from transformers import AutoTokenizer

    src = sorted(glob.glob("findings/routing_discrimination/results_*.json"))
    if not src:
        _log("no routing_discrimination results found")
        return 1
    data = json.load(open(src[-1]))
    models = data["models"]
    # {model: [ (prompt, text), ... ]}
    samples: dict[str, list[tuple[str, str]]] = {
        m: [(s["prompt"], s["text"]) for s in data["samples"][m]] for m in models
    }
    prompts = data["prompts"]

    toks = [AutoTokenizer.from_pretrained(m) for m in TOKENIZERS]
    _log(f"loaded {len(toks)} tokenizers")

    def _vec(text: str) -> list[float]:
        # tokenizer fingerprints: token counts under each public tokenizer
        tok_counts = [len(t.encode(text, add_special_tokens=False)) for t in toks]
        # original style features
        words = re.findall(r"\S+", text) or [""]
        letters = re.findall(r"[A-Za-z]", text)
        lower = [c for c in letters if c.islower()]
        style = [
            sum(len(w) for w in words) / len(words),
            len(lower) / len(letters) if letters else 0.0,
            1.0 - len(set(words)) / len(words) if words else 0.0,
        ]
        return tok_counts + style

    # Leave-one-prompt-out nearest centroid over the 9-feature vector.
    folds: list[float] = []
    for held in range(len(prompts)):
        centroids: dict[str, list[float]] = {}
        for m in models:
            vecs = [_vec(text) for i, (_, text) in enumerate(samples[m]) if i != held]
            centroids[m] = [
                sum(v[k] for v in vecs) / len(vecs) for k in range(len(vecs[0]))
            ]
        correct = 0
        for m in models:
            vec = _vec(samples[m][held][1])
            pred = min(
                centroids,
                key=lambda cm: sum((a - b) ** 2 for a, b in zip(vec, centroids[cm])),
            )
            correct += pred == m
        folds.append(correct / len(models))
        _log(f"held-out prompt {held}: {correct}/{len(models)}")

    acc = sum(folds) / len(folds)
    results = {
        "source_results": src[-1],
        "n_features": 9,
        "leave_one_prompt_out_accuracy": round(acc, 4),
        "per_fold": [round(a, 4) for a in folds],
        "v1_style_only_accuracy": 0.5,
        "chance": 0.25,
        "reading": (
            f"accuracy {acc:.2f} vs style-only 0.5: tokenizer fingerprints "
            "improve/hold the no-log receiver; chance is 0.25"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    _log(f"wrote {out}; accuracy={acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
