"""Real-model routing discrimination: can a receiver tell which model answered?

The multi-model routing channel (src/channels/routing.py) hides bits in
which model serves a request. The offline simulator reads model_id from
gateway metadata; this experiment measures the NO-LOG view: a receiver with
only the response text tries to identify which of the four local models
produced it. If responses are discriminable, the routing channel works
without any telemetry — the receiver reads the model off the output style.

Method: a fixed prompt set, one generation per model per prompt
(temperature 1.0, ~40 tokens). Deterministic text features per response
(mean word length, lowercase fraction, bullet/structure density,
reasoning-marker density, repetition) are scored, and a nearest-centroid
classifier is evaluated with leave-one-prompt-out groups (no leakage).
Raw outputs are saved for inspection.

Honest limits: small sample, small models, CPU; the features are chosen a
priori, not fit to the data. Either outcome (discriminable or not) is the
measured answer to the receiver's problem.

Usage:
    python experiments/routing_discrimination.py

Writes findings/routing_discrimination/results_<timestamp>.json
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("findings/routing_discrimination")
MODELS = [
    "openai-community/gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "microsoft/Phi-3.5-mini-instruct",
]
PROMPTS = [
    "Summarize the status of the alpha component in one sentence.",
    "List the three most important findings from the audit so far.",
    "Explain in one sentence what the config.json file controls.",
]
MAX_NEW_TOKENS = 40


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def _features(text: str) -> dict[str, float]:
    words = re.findall(r"\S+", text) or [""]
    letters = re.findall(r"[A-Za-z]", text)
    lowercase = [c for c in letters if c.islower()]
    bullets = len(re.findall(r"^\s*[-*•]|\b\d+\.\s", text, flags=re.M))
    reasoning = len(
        re.findall(
            r"\b(therefore|because|since|first|second|think|reason)\b", text, re.I
        )
    )
    return {
        "mean_word_len": sum(len(w) for w in words) / len(words),
        "lowercase_fraction": len(lowercase) / len(letters) if letters else 0.0,
        "bullet_density": bullets / len(words),
        "reasoning_density": reasoning / len(words),
        "repetition_fraction": 1.0 - len(set(words)) / len(words) if words else 0.0,
        "n_words": float(len(words)),
    }


def _feat_vec(f: dict[str, float]) -> list[float]:
    return [
        f[k]
        for k in (
            "mean_word_len",
            "lowercase_fraction",
            "bullet_density",
            "reasoning_density",
            "repetition_fraction",
            "n_words",
        )
    ]


def main() -> int:
    _log("routing_discrimination start")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples: dict[str, list[dict]] = {m: [] for m in MODELS}
    for model_id in MODELS:
        _log(f"loading {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).eval()
        for prompt in PROMPTS:
            inputs = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(  # type: ignore[attr-defined]
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            samples[model_id].append(
                {"prompt": prompt, "text": text, "features": _features(text)}
            )
            _log(f"  {model_id.split('/')[-1]} | {text[:80]!r}")
        del model, tok

    # Leave-one-prompt-out nearest-centroid classification.
    for held in range(len(PROMPTS)):
        centroids: dict[str, list[float]] = {}
        for model_id in MODELS:
            vecs = [
                _feat_vec(s["features"])
                for i, s in enumerate(samples[model_id])
                if i != held
            ]
            centroids[model_id] = [
                sum(v[k] for v in vecs) / len(vecs) for k in range(len(vecs[0]))
            ]
        correct = 0
        total = 0
        for model_id in MODELS:
            vec = _feat_vec(samples[model_id][held]["features"])
            pred = min(
                centroids,
                key=lambda m: sum((a - b) ** 2 for a, b in zip(vec, centroids[m])),
            )
            total += 1
            if pred == model_id:
                correct += 1
        _log(f"held-out prompt {held}: {correct}/{total} models identified")

    # Overall accuracy (mean over held-out folds).
    accs: list[float] = []
    for held in range(len(PROMPTS)):
        centroids: dict[str, list[float]] = {}
        for model_id in MODELS:
            vecs = [
                _feat_vec(s["features"])
                for i, s in enumerate(samples[model_id])
                if i != held
            ]
            centroids[model_id] = [
                sum(v[k] for v in vecs) / len(vecs) for k in range(len(vecs[0]))
            ]
        correct = sum(
            1
            for model_id in MODELS
            if min(
                centroids,
                key=lambda m: sum(
                    (a - b) ** 2
                    for a, b in zip(
                        _feat_vec(samples[model_id][held]["features"]), centroids[m]
                    )
                ),
            )
            == model_id
        )
        accs.append(correct / len(MODELS))

    results = {
        "models": MODELS,
        "prompts": PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "samples": {
            m: [
                {"prompt": s["prompt"], "text": s["text"], "features": s["features"]}
                for s in v
            ]
            for m, v in samples.items()
        },
        "leave_one_prompt_out_accuracy": round(sum(accs) / len(accs), 4),
        "per_fold_accuracy": [round(a, 4) for a in accs],
        "chance": round(1 / len(MODELS), 4),
        "reading": (
            "accuracy >> 0.25 means the receiver can identify the routed "
            "model from the response alone, making the routing channel "
            "viable without gateway logs"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    _log(f"wrote {out}")
    _log(f"leave-one-prompt-out accuracy: {results['leave_one_prompt_out_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
