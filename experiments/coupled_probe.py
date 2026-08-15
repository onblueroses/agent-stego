"""Activation-probe measurements under frozen distribution-coupling cells.

Previous probe experiment used greedy normal baselines, creating an
artificial gap. This experiment matches sampling conditions:
- Normal: stochastic sample from coupled distribution (temp + top_p)
- Stego: arithmetic coding over the same coupled distribution

The output remains a checkpoint until a powered design includes a same-rig
positive control and preregistered decision threshold.

Usage:
    MODEL=Qwen/Qwen2.5-3B python experiments/coupled_probe.py
    MODEL=Qwen/Qwen2.5-7B python experiments/coupled_probe.py
"""

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import torch

from src.experiment_inputs import synthetic_text
from src.token_stego.activation_probe import evaluate_probe, train_probe
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.experiment_design import (
    assert_no_cross_split_duplicates,
    grouped_train_test_indices,
    select_unique_prompts,
    split_manifest,
)
from src.token_stego.latent import LatentCapture, LatentSnapshot
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/coupled_probe")


def _key_label(key: bytes | None) -> str:
    """Redacted label: presence only, no key-derived data."""
    return "none" if key is None else "set"


# Frozen historical replay cells: raw, moderate, and aggressive. This grid is
# not evidence that the coupling space has been covered.
COUPLING_GRID = [
    {"temperature": 1.0, "top_p": 1.0},  # Raw (no coupling) - baseline
    {"temperature": 0.9, "top_p": 0.9},  # Moderate combined coupling
    {"temperature": 0.7, "top_p": 0.85},  # Aggressive coupling
]

# Frozen replay identifiers from the corrected three-seed protocol. They are not
# a claim that three seeds capture population uncertainty.
PROBE_SEEDS = (42, 314, 2718)
# The prior experiment trained for 200 full-batch epochs. This replay value is
# not a convergence claim; a replacement protocol needs a held-out stop rule.
PROBE_TRAINING_EPOCHS = 200


def stochastic_sample(dist: list[float], rng: random.Random) -> int:
    """Sample a token from a probability distribution."""
    if not dist or any(
        not isinstance(p, (int, float))
        or isinstance(p, bool)
        or not math.isfinite(p)
        or p < 0
        for p in dist
    ):
        raise ValueError("Sampling distribution must be finite and non-negative")
    total = sum(dist)
    if total <= 0:
        raise ValueError("Sampling distribution must have positive mass")
    r = rng.random() * total
    cumsum = 0.0
    for i, p in enumerate(dist):
        cumsum += p
        if cumsum >= r:
            return i
    return len(dist) - 1


def generate_normal_stochastic(
    model: StegoModel,
    prompt: str,
    n_tokens: int,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> list[int]:
    """Generate tokens via stochastic sampling with coupling params."""
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 0:
        raise ValueError("n_tokens must be a non-negative integer")
    context = model.tokenize(prompt)
    tokens: list[int] = []
    past_kv = None
    for _ in range(n_tokens):
        dist, past_kv = model.get_distribution(
            context + tokens,
            temperature,
            top_p,
            past_key_values=past_kv,
            use_cache=True,
        )
        tokens.append(stochastic_sample(dist, rng))
    return tokens


def compute_entropy_auc(
    normal_entropies: list[float], stego_entropies: list[float]
) -> float:
    """Compute oriented AUC from entropy lists (higher = more detectable)."""
    if not normal_entropies or not stego_entropies:
        raise ValueError("Entropy AUC requires both normal and stego observations")
    labels = [0] * len(normal_entropies) + [1] * len(stego_entropies)
    scores = normal_entropies + stego_entropies

    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    i = 0
    while i < len(pairs):
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
        prev_fpr, prev_tpr = fpr, tpr
        i = j

    # Return oriented AUC (always >= 0.5)
    return max(auc, 1 - auc)


def mean_token_entropy(
    model: StegoModel,
    tokens: list[int],
    context_ids: list[int],
    temperature: float,
    top_p: float,
) -> float:
    """Compute mean per-token entropy for a generated sequence."""
    current = list(context_ids)
    entropies = []
    past_kv = None
    for tok in tokens:
        dist, past_kv = model.get_distribution(
            current,
            temperature,
            top_p,
            past_key_values=past_kv,
            use_cache=True,
        )
        p = dist[tok]
        if p <= 0:
            raise ValueError("Generated token has zero probability under its model")
        entropies.append(-math.log(p))
        current.append(tok)
    return sum(entropies) / len(entropies) if entropies else 0.0


def run_single_condition(
    model: StegoModel,
    temperature: float,
    top_p: float,
    n_samples: int,
    max_tokens: int,
    secret: str,
    key: bytes | None,
) -> dict:
    """Run probe experiment for one coupling configuration."""
    key_label = _key_label(key)
    print(f"\n--- T={temperature}, top_p={top_p}, key={key_label} ---")

    rng = random.Random(42)
    channel = ArithmeticStegoChannel(
        model, key=key, temperature=temperature, top_p=top_p
    )
    sample_prompts = select_unique_prompts(n_samples)
    split_seed = 42
    train_idx, test_idx = grouped_train_test_indices(sample_prompts, seed=split_seed)

    normal_snaps: list[LatentSnapshot] = []
    stego_snaps: list[LatentSnapshot] = []
    normal_entropies: list[float] = []
    stego_entropies: list[float] = []
    stego_lengths: list[int] = []

    # Stego first - its actual length determines the matched-normal length.
    print("  Stego (coupled arithmetic coding)...")
    stego_runs: list[tuple[str, list[int], list[int]]] = []
    for prompt in sample_prompts:
        context_ids = model.tokenize(prompt)
        output = channel.encode(secret, prompt, max_tokens=max_tokens)
        stego_runs.append((prompt, context_ids, output.tokens))
        stego_lengths.append(len(output.tokens))

        with LatentCapture(model) as cap:
            full_ids = context_ids + output.tokens
            cap.clear()
            model.get_distribution(full_ids[:-1], temperature, top_p)
            stego_snaps.append(cap.snapshot())

        ent = mean_token_entropy(model, output.tokens, context_ids, temperature, top_p)
        stego_entropies.append(ent)

    # Normal: per-pair length-matched to stego sample i.
    print("  Normal (stochastic, length-matched)...")
    normal_runs: list[tuple[str, list[int], list[int]]] = []
    for prompt, context_ids, stego_toks in stego_runs:
        n_tok = len(stego_toks)
        tokens = generate_normal_stochastic(
            model, prompt, n_tok, temperature, top_p, rng
        )
        normal_runs.append((prompt, context_ids, tokens))

        with LatentCapture(model) as cap:
            full_ids = context_ids + tokens
            model.get_distribution(full_ids[:-1], temperature, top_p)
            normal_snaps.append(cap.snapshot())

        ent = mean_token_entropy(model, tokens, context_ids, temperature, top_p)
        normal_entropies.append(ent)

    # Train and evaluate activation probe
    print("  Training probe...")
    normal_observations = [
        (prompt, tuple(context_ids), tuple(tokens))
        for prompt, context_ids, tokens in normal_runs
    ]
    stego_observations = [
        (prompt, tuple(context_ids), tuple(tokens))
        for prompt, context_ids, tokens in stego_runs
    ]
    assert_no_cross_split_duplicates(normal_observations, train_idx, test_idx)
    assert_no_cross_split_duplicates(stego_observations, train_idx, test_idx)

    train_normal = [normal_snaps[i] for i in train_idx]
    train_stego = [stego_snaps[i] for i in train_idx]
    test_normal = [normal_snaps[i] for i in test_idx]
    test_stego = [stego_snaps[i] for i in test_idx]

    test_snaps = test_normal + test_stego
    test_labels = [0] * len(test_normal) + [1] * len(test_stego)
    test_groups = [sample_prompts[i] for i in test_idx]
    seed_metrics = []
    for probe_seed in PROBE_SEEDS:
        probe = train_probe(
            train_normal,
            train_stego,
            epochs=PROBE_TRAINING_EPOCHS,
            seed=probe_seed,
        )
        run_metrics = evaluate_probe(
            probe,
            test_snaps,
            test_labels,
            seed=probe_seed,
            group_ids=test_groups + test_groups,
        )
        seed_metrics.append(
            {
                "seed": probe_seed,
                "auc": run_metrics["auc"],
                "auc_ci_95": run_metrics["auc_ci_95"],
                "accuracy": run_metrics["accuracy"],
            }
        )
    probe_auc = sum(item["auc"] for item in seed_metrics) / len(seed_metrics)
    probe_accuracy = sum(item["accuracy"] for item in seed_metrics) / len(seed_metrics)
    probe_auc_seed_ci_envelope = [
        min(item["auc_ci_95"][0] for item in seed_metrics),
        max(item["auc_ci_95"][1] for item in seed_metrics),
    ]

    # Token-level entropy AUC
    entropy_auc = compute_entropy_auc(
        [normal_entropies[i] for i in test_idx],
        [stego_entropies[i] for i in test_idx],
    )

    result = {
        "temperature": temperature,
        "top_p": top_p,
        "key": key_label,
        "probe_auc": probe_auc,
        "probe_auc_seed_ci_envelope": probe_auc_seed_ci_envelope,
        "probe_accuracy": probe_accuracy,
        "probe_seed_runs": seed_metrics,
        "entropy_auc": entropy_auc,
        "normal_entropy_mean": sum(normal_entropies[i] for i in test_idx)
        / len(test_idx),
        "stego_entropy_mean": sum(stego_entropies[i] for i in test_idx) / len(test_idx),
        "entropy_gap": abs(
            sum(normal_entropies[i] for i in test_idx) / len(test_idx)
            - sum(stego_entropies[i] for i in test_idx) / len(test_idx)
        ),
        "stego_len_mean": sum(stego_lengths[i] for i in test_idx) / len(test_idx),
        "stego_len_min": min(stego_lengths[i] for i in test_idx),
        "stego_len_max": max(stego_lengths[i] for i in test_idx),
        "n_samples": n_samples,
        "split": split_manifest(sample_prompts, train_idx, test_idx, seed=split_seed),
    }

    print(
        f"  Probe AUC: {probe_auc:.3f} "
        f"(per-seed bootstrap envelope "
        f"{probe_auc_seed_ci_envelope[0]:.3f}-"
        f"{probe_auc_seed_ci_envelope[1]:.3f})"
    )
    print(f"  Probe accuracy: {probe_accuracy:.3f}")
    print(f"  Entropy AUC: {entropy_auc:.3f}")
    print(
        f"  Entropy gap: {result['entropy_gap']:.3f} "
        f"(normal={result['normal_entropy_mean']:.3f}, "
        f"stego={result['stego_entropy_mean']:.3f})"
    )

    return result


def run_experiment() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    # Historical replay defaults only. Statistical power and the token ceiling
    # must be chosen explicitly before treating a future run as evidence.
    n_samples = int(os.environ.get("N", "20"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "50"))
    # This historical replay payload is longer than the earlier short fixture;
    # its length is not a capacity claim.
    secret = synthetic_text(
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG IN THE FIELD AT NIGHT",
    )
    key_env = os.environ.get("KEY", "probe-test")
    key: bytes | None = (
        None if key_env.lower() in ("", "none", "off") else key_env.encode("utf-8")
    )
    # Optional coupling-cell filter: comma-separated "T,p" pairs. Empty = full grid.
    # e.g. COUPLING_FILTER="0.7,0.85" runs only the aggressive cell.
    filt = os.environ.get("COUPLING_FILTER", "").strip()
    grid: list[dict[str, float]] = list(COUPLING_GRID)
    if filt:
        wanted = {tuple(float(x) for x in pair.split(",")) for pair in filt.split(";")}
        grid = [g for g in COUPLING_GRID if (g["temperature"], g["top_p"]) in wanted]
        if not grid:
            raise SystemExit(f"COUPLING_FILTER={filt!r} matched no grid cells")

    # Historical Qwen runs observed NaNs with float16 and used bfloat16 instead.
    dtype = torch.bfloat16
    print(f"Model: {model_name}")
    print(f"Samples per condition: {n_samples} normal + {n_samples} stego")
    print(f"Payload length: {len(secret.encode('ascii')) * 8} bits")
    print(f"Max tokens: {max_tokens}")
    print(f"Key: {_key_label(key)}")
    print(f"Coupling grid: {len(COUPLING_GRID)} conditions")
    print(f"dtype: {dtype}")
    print()

    model = StegoModel(model_name, dtype=dtype)
    print("Loading model...")
    model.load()
    print("Model loaded.\n")

    results = []
    for params in grid:
        result = run_single_condition(
            model=model,
            temperature=params["temperature"],
            top_p=params["top_p"],
            n_samples=n_samples,
            max_tokens=max_tokens,
            secret=secret,
            key=key,
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Temp':>5} {'TopP':>5} {'Probe AUC':>10} {'Ent AUC':>10} {'Ent Gap':>10}")
    print("-" * 45)
    for r in results:
        print(
            f"{r['temperature']:>5.1f} {r['top_p']:>5.2f} "
            f"{r['probe_auc']:>10.3f} {r['entropy_auc']:>10.3f} "
            f"{r['entropy_gap']:>10.3f}"
        )

    # Descriptive checkpoint only. A verdict requires a preregistered threshold,
    # a powered design, and a positive control in the same rig.
    lowest_observed = min(results, key=lambda r: r["probe_auc"])
    print(
        f"\nLowest observed probe AUC: {lowest_observed['probe_auc']:.3f} "
        f"at T={lowest_observed['temperature']}, "
        f"top_p={lowest_observed['top_p']}"
    )
    print(
        "VERDICT: not yet established; power, a decision threshold, and a "
        "same-rig positive control are unspecified."
    )
    print(
        "UNTRIED: coupling values outside the frozen grid, other model families, "
        "other task families, and other data regimes."
    )
    print(
        "NEXT ROUTE: preregister the detectable effect and positive control, size "
        "independent prompt groups, then rerun the frozen split."
    )

    # Save
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "payload_bits": len(secret.encode("ascii")) * 8,
                "payload_retained": False,
                "key": _key_label(key),
                "length_match": "per-pair (normal matched to stego length)",
                "n_samples": n_samples,
                "max_tokens": max_tokens,
                "dtype": str(dtype),
                "results": results,
                "lowest_observed_probe_auc": lowest_observed["probe_auc"],
                "lowest_observed_params": {
                    "temperature": lowest_observed["temperature"],
                    "top_p": lowest_observed["top_p"],
                },
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
