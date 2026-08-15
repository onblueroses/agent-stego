"""Detection experiment for token-likelihood steganography.

Generates stego and normal texts, runs detector on both,
computes ROC curve and AUC.

Usage:
    MODEL=Qwen/Qwen2.5-3B python experiments/token_detection.py
"""

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import torch

from src.experiment_inputs import synthetic_text
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.detector import StegoDetector
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/token_likelihood")

PROMPTS = [
    "The history of computing began with",
    "In recent years, advances in machine learning have",
    "The weather forecast for tomorrow indicates",
    "A comprehensive review of the literature shows",
    "Modern software engineering practices emphasize",
    "The development of quantum computing has",
    "Recent studies in neuroscience suggest that",
    "The economic impact of automation on",
]


def generate_normal_text(model: StegoModel, prompt: str, max_tokens: int) -> str:
    """Generate text via stochastic sampling (no steganography).

    Uses the same default temperature/top_p as the stego channel so
    the baseline distribution matches what arithmetic coding competes against.
    """
    context_ids = model.tokenize(prompt)
    generated: list[int] = []
    past_kv = None

    for _ in range(max_tokens):
        dist, past_kv = model.get_distribution(
            context_ids + generated,
            past_key_values=past_kv,
            use_cache=True,
        )
        # Stochastic sampling to match stego channel distribution
        token_id = random.choices(range(len(dist)), weights=dist)[0]
        generated.append(token_id)

        # Stop on EOS
        if (
            hasattr(model.tokenizer, "eos_token_id")
            and token_id == model.tokenizer.eos_token_id
        ):
            break

    return model.detokenize(generated)


def _validate_binary_inputs(labels: list[int], scores: list[float]) -> tuple[int, int]:
    """Validate the two-class sample required by rank-based metrics."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have equal length")
    if any(isinstance(label, bool) or label not in (0, 1) for label in labels):
        raise ValueError("labels must contain only integer class ids 0 and 1")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        raise ValueError("binary metrics require observations from both classes")
    return total_pos, total_neg


def compute_auc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC from labels (0=normal, 1=stego) and scores."""
    total_pos, total_neg = _validate_binary_inputs(labels, scores)
    # Sort by score only (not label) to avoid tie-breaking bias
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    fp = 0

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    i = 0
    while i < len(pairs):
        # Process all tied scores together
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
        i = j

    return auc


def find_rate_at_threshold(
    labels: list[int], scores: list[float], target_fpr: float
) -> float:
    """Return the highest tie-safe TPR whose empirical FPR is at most the target."""
    total_pos, total_neg = _validate_binary_inputs(labels, scores)
    if not math.isfinite(target_fpr) or not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be finite and in the interval [0, 1]")
    pairs = sorted(zip(scores, labels), key=lambda item: -item[0])
    tp = 0
    fp = 0
    index = 0
    while index < len(pairs):
        end = index
        tied_pos = 0
        tied_neg = 0
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            if pairs[end][1] == 1:
                tied_pos += 1
            else:
                tied_neg += 1
            end += 1
        if (fp + tied_neg) / total_neg > target_fpr:
            break
        tp += tied_pos
        fp += tied_neg
        index = end
    return tp / total_pos


def run_detection() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    n_samples = int(os.environ.get("N", str(len(PROMPTS))))
    max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
    secret = synthetic_text("HI")
    if not 0 < n_samples <= len(PROMPTS):
        raise ValueError(f"N must be between 1 and the {len(PROMPTS)} distinct prompts")
    if max_tokens <= 0:
        raise ValueError("MAX_TOKENS must be positive")

    dtype = torch.float32
    if "7B" in model_name or "7b" in model_name:
        dtype = torch.float16

    print(f"Model: {model_name}")
    print(f"Samples: {n_samples} stego + {n_samples} normal")
    print(f"Synthetic payload length: {len(secret.encode('utf-8')) * 8} bits")
    print()

    print("Loading model...")
    model = StegoModel(model_name, dtype=dtype)
    model.load()

    channel = ArithmeticStegoChannel(model)
    detector = StegoDetector(model)

    labels: list[int] = []
    entropy_scores: list[float] = []
    cdf_scores: list[float] = []
    combined_scores: list[float] = []

    # Generate stego texts
    print("Generating stego texts...")
    for i in range(n_samples):
        prompt = PROMPTS[i]
        output = channel.encode(secret, prompt, max_tokens=max_tokens)

        result = detector.score(output.text, prompt)
        labels.append(1)
        entropy_scores.append(detector.score_entropy(output.text, prompt).score)
        cdf_scores.append(detector.score_cdf_uniformity(output.text, prompt).score)
        combined_scores.append(result.score)

        print(
            f"  Stego {i + 1}/{n_samples}: score={result.score:.3f} p={result.p_value:.3f}"
        )

    # Generate normal texts
    print("\nGenerating normal texts...")
    for i in range(n_samples):
        prompt = PROMPTS[i]
        text = generate_normal_text(model, prompt, max_tokens)

        result = detector.score(text, prompt)
        labels.append(0)
        entropy_scores.append(detector.score_entropy(text, prompt).score)
        cdf_scores.append(detector.score_cdf_uniformity(text, prompt).score)
        combined_scores.append(result.score)

        print(
            f"  Normal {i + 1}/{n_samples}: score={result.score:.3f} p={result.p_value:.3f}"
        )

    # Compute ROC metrics
    auc_entropy = compute_auc(labels, entropy_scores)
    auc_cdf = compute_auc(labels, cdf_scores)
    auc_combined = compute_auc(labels, combined_scores)

    print(f"\n{'=' * 60}")
    print("Detection Results")
    print(f"{'=' * 60}")
    print(f"AUC (entropy):   {auc_entropy:.3f}")
    print(f"AUC (CDF):       {auc_cdf:.3f}")
    print(f"AUC (combined):  {auc_combined:.3f}")
    print(
        "Interpretation requires a preregistered decision threshold and power analysis."
    )
    print(f"{'=' * 60}")

    # Save results
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"detection_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "payload_bits": len(secret.encode("utf-8")) * 8,
                "n_samples": n_samples,
                "max_tokens": max_tokens,
                "auc_entropy": auc_entropy,
                "auc_cdf": auc_cdf,
                "auc_combined": auc_combined,
                "labels": labels,
                "entropy_scores": entropy_scores,
                "cdf_scores": cdf_scores,
                "combined_scores": combined_scores,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_detection()
