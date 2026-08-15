"""Capacity-detection Pareto frontier + probe transfer experiment.

Two questions in one run:

1. CAPACITY vs DETECTION: What net recovered bits/token and probe scores appear
   in each frozen replay cell?

2. PROBE TRANSFER: How does a raw-trained probe score group-disjoint samples
   from the frozen coupling cells? The driver does not turn those scores into a
   transfer verdict without a powered design and positive control.

Usage:
    MODEL=Qwen/Qwen2.5-3B python experiments/capacity_and_transfer.py
    MODEL=Qwen/Qwen2.5-7B python experiments/capacity_and_transfer.py
"""

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from src.experiment_inputs import synthetic_text
from src.token_stego.activation_probe import evaluate_probe, train_probe
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.experiment_design import (
    PROBE_PROMPTS,
    assert_no_cross_split_duplicates,
    grouped_train_test_indices,
    select_unique_prompts,
    split_manifest,
)
from src.token_stego.latent import LatentCapture, LatentSnapshot
from src.token_stego.model import StegoModel

FINDINGS_DIR = Path("findings/capacity_transfer")


def _key_label(key: bytes | None) -> str:
    """Redacted label: presence only, no key-derived data."""
    return "none" if key is None else "set"


SECRETS = [
    "HI",  # 16 bits
    "HELLO",  # 40 bits
    "SECRET MSG",  # 80 bits
    "THE KEY IS 42",  # 104 bits
]

# Frozen historical replay cells: raw, moderate, and aggressive. This grid is
# not evidence that the coupling space has been covered.
COUPLING_GRID = [
    {"temperature": 1.0, "top_p": 1.0, "label": "raw"},
    {"temperature": 0.9, "top_p": 0.9, "label": "moderate"},
    {"temperature": 0.7, "top_p": 0.85, "label": "aggressive"},
]

# Frozen replay identifiers from the corrected three-seed protocol. They are not
# a claim that three seeds capture population uncertainty.
PROBE_SEEDS = (42, 314, 2718)
# The prior experiment trained for 200 full-batch epochs. This replay value is
# not a convergence claim; a replacement protocol needs a held-out stop rule.
PROBE_TRAINING_EPOCHS = 200


@dataclass
class ProbeDataset:
    normal: list[LatentSnapshot]
    stego: list[LatentSnapshot]
    prompts: list[str]
    normal_observations: list[tuple[str, tuple[int, ...], tuple[int, ...]]]
    stego_observations: list[tuple[str, tuple[int, ...], tuple[int, ...]]]


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


def measure_capacity(
    model: StegoModel,
    temperature: float,
    top_p: float,
    secrets: list[str],
    n_trials: int,
    max_tokens: int,
    key: bytes | None,
) -> dict:
    """Measure bits/token across message lengths for one coupling config."""
    channel = ArithmeticStegoChannel(
        model, key=key, temperature=temperature, top_p=top_p
    )

    results = []
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("Capacity measurement requires at least one trial")
    if not secrets:
        raise ValueError("Capacity measurement requires at least one secret")
    if n_trials > len(PROBE_PROMPTS):
        raise ValueError(
            f"Capacity trials exceed the {len(PROBE_PROMPTS)} unique prompts"
        )
    for payload_index, secret in enumerate(secrets, start=1):
        secret_bits = len(secret.encode("ascii")) * 8
        attempted_bpt_samples = []
        committed_bpt_samples = []
        net_bpt_samples = []
        recovery_samples = []

        for trial in range(n_trials):
            prompt = PROBE_PROMPTS[trial]
            output = channel.encode(secret, prompt, max_tokens=max_tokens)
            attempted_bpt = (
                secret_bits / output.tokens_generated
                if output.tokens_generated > 0
                else 0
            )
            committed_bpt = (
                output.bits_encoded / output.tokens_generated
                if output.tokens_generated > 0
                else 0
            )
            attempted_bpt_samples.append(attempted_bpt)
            committed_bpt_samples.append(committed_bpt)

            # Check recovery
            metrics = channel.measure(secret, output.text, prompt, nonce=output.nonce)
            recovery_samples.append(metrics.exact_recovery)
            net_bpt_samples.append(
                secret_bits / output.tokens_generated
                if output.tokens_generated > 0 and metrics.exact_recovery
                else 0
            )

        results.append(
            {
                "payload_id": f"payload-{payload_index}",
                "secret_bits": secret_bits,
                "mean_attempted_bpt": sum(attempted_bpt_samples)
                / len(attempted_bpt_samples),
                "mean_committed_bpt": sum(committed_bpt_samples)
                / len(committed_bpt_samples),
                "mean_net_bpt": sum(net_bpt_samples) / len(net_bpt_samples),
                "min_net_bpt": min(net_bpt_samples),
                "max_net_bpt": max(net_bpt_samples),
                "recovery_rate": sum(recovery_samples) / len(recovery_samples),
                "n_trials": n_trials,
            }
        )

    return {
        "capacity_by_message": results,
        "overall_mean_net_bpt": sum(r["mean_net_bpt"] for r in results) / len(results),
    }


def collect_snapshots(
    model: StegoModel,
    temperature: float,
    top_p: float,
    prompts: list[str],
    max_tokens: int,
    secret: str,
    key: bytes | None,
) -> ProbeDataset:
    """Collect normal and stego activation snapshots, length-matched per pair."""
    rng = random.Random(42)
    channel = ArithmeticStegoChannel(
        model, key=key, temperature=temperature, top_p=top_p
    )

    normal_snaps: list[LatentSnapshot] = []
    stego_snaps: list[LatentSnapshot] = []
    stego_runs: list[tuple[str, list[int], list[int]]] = []

    # Stego first - its length determines the matched-normal length.
    for prompt in prompts:
        context_ids = model.tokenize(prompt)
        output = channel.encode(secret, prompt, max_tokens=max_tokens)
        stego_runs.append((prompt, context_ids, output.tokens))

        with LatentCapture(model) as cap:
            full_ids = context_ids + output.tokens
            cap.clear()
            model.get_distribution(full_ids[:-1], temperature, top_p)
            stego_snaps.append(cap.snapshot())

    # Normal length-matched to stego sample i.
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

    return ProbeDataset(
        normal=normal_snaps,
        stego=stego_snaps,
        prompts=list(prompts),
        normal_observations=[
            (prompt, tuple(context_ids), tuple(tokens))
            for prompt, context_ids, tokens in normal_runs
        ],
        stego_observations=[
            (prompt, tuple(context_ids), tuple(tokens))
            for prompt, context_ids, tokens in stego_runs
        ],
    )


def train_and_evaluate(
    train_normal: list[LatentSnapshot],
    train_stego: list[LatentSnapshot],
    test_normal: list[LatentSnapshot],
    test_stego: list[LatentSnapshot],
    test_groups: list[str],
) -> dict:
    """Train seeded probes and return uncertainty-aware aggregate metrics."""
    test_snaps = test_normal + test_stego
    test_labels = [0] * len(test_normal) + [1] * len(test_stego)
    seed_runs = []
    for probe_seed in PROBE_SEEDS:
        probe = train_probe(
            train_normal,
            train_stego,
            epochs=PROBE_TRAINING_EPOCHS,
            seed=probe_seed,
        )
        metrics = evaluate_probe(
            probe,
            test_snaps,
            test_labels,
            seed=probe_seed,
            group_ids=test_groups + test_groups,
        )
        seed_runs.append(
            {
                "seed": probe_seed,
                "auc": metrics["auc"],
                "auc_ci_95": metrics["auc_ci_95"],
                "accuracy": metrics["accuracy"],
            }
        )
    return {
        "auc": sum(item["auc"] for item in seed_runs) / len(seed_runs),
        "auc_seed_ci_envelope": [
            min(item["auc_ci_95"][0] for item in seed_runs),
            max(item["auc_ci_95"][1] for item in seed_runs),
        ],
        "accuracy": sum(item["accuracy"] for item in seed_runs) / len(seed_runs),
        "seed_runs": seed_runs,
    }


def run_experiment() -> None:
    model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-3B")
    # Historical replay defaults only. Statistical power and the token ceiling
    # must be chosen explicitly before treating a future run as evidence.
    n_samples = int(os.environ.get("N", "20"))
    n_capacity_trials = int(os.environ.get("CAPACITY_TRIALS", "5"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "50"))
    probe_secret = synthetic_text(
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG IN THE FIELD AT NIGHT",
    )
    key_env = os.environ.get("KEY", "transfer-test")
    key: bytes | None = (
        None if key_env.lower() in ("", "none", "off") else key_env.encode("utf-8")
    )

    dtype = torch.bfloat16
    print(f"Model: {model_name}")
    print(f"Probe samples: {n_samples} per condition")
    print(f"Capacity trials: {n_capacity_trials} per message per condition")
    print(f"Max tokens: {max_tokens}")
    print(f"Probe payload length: {len(probe_secret.encode('ascii')) * 8} bits")
    print(f"Key: {_key_label(key)}")
    print(f"Coupling grid: {len(COUPLING_GRID)} conditions")
    print()

    model = StegoModel(model_name, dtype=dtype)
    print("Loading model...")
    model.load()
    print("Model loaded.\n")

    # ===== PART 1: Capacity-Detection Pareto =====
    print("=" * 60)
    print("PART 1: CAPACITY-DETECTION PARETO FRONTIER")
    print("=" * 60)

    pareto_results = []
    all_snapshots: dict[str, ProbeDataset] = {}
    sample_prompts = select_unique_prompts(n_samples)
    split_seed = 42
    train_idx, test_idx = grouped_train_test_indices(sample_prompts, seed=split_seed)

    for params in COUPLING_GRID:
        label = params["label"]
        temp = params["temperature"]
        top_p = params["top_p"]
        print(f"\n--- {label} (T={temp}, p={top_p}) ---")

        # Capacity measurement
        print("  Measuring capacity...")
        cap_result = measure_capacity(
            model, temp, top_p, SECRETS, n_capacity_trials, max_tokens, key
        )

        # Collect snapshots for probe (reused in transfer experiment)
        print("  Collecting activation snapshots...")
        dataset = collect_snapshots(
            model, temp, top_p, sample_prompts, max_tokens, probe_secret, key
        )
        assert_no_cross_split_duplicates(
            dataset.normal_observations, train_idx, test_idx
        )
        assert_no_cross_split_duplicates(
            dataset.stego_observations, train_idx, test_idx
        )
        all_snapshots[label] = dataset

        # Train and evaluate probe (same-condition)
        print("  Training probe...")
        metrics = train_and_evaluate(
            [dataset.normal[i] for i in train_idx],
            [dataset.stego[i] for i in train_idx],
            [dataset.normal[i] for i in test_idx],
            [dataset.stego[i] for i in test_idx],
            [sample_prompts[i] for i in test_idx],
        )

        result = {
            "label": label,
            "temperature": temp,
            "top_p": top_p,
            "mean_net_bpt": cap_result["overall_mean_net_bpt"],
            "probe_auc": metrics["auc"],
            "probe_auc_seed_ci_envelope": metrics["auc_seed_ci_envelope"],
            "probe_accuracy": metrics["accuracy"],
            "probe_seed_runs": metrics["seed_runs"],
            "capacity_detail": cap_result["capacity_by_message"],
        }
        pareto_results.append(result)

        print(f"  Net bits/token: {cap_result['overall_mean_net_bpt']:.2f}")
        print(
            f"  Probe AUC: {metrics['auc']:.3f} "
            f"(per-seed bootstrap envelope "
            f"{metrics['auc_seed_ci_envelope'][0]:.3f}-"
            f"{metrics['auc_seed_ci_envelope'][1]:.3f})"
        )
        for cr in cap_result["capacity_by_message"]:
            print(
                f"    {cr['payload_id']:>14s} ({cr['secret_bits']:>3d}b): "
                f"{cr['mean_net_bpt']:.2f} net bpt, "
                f"recovery {cr['recovery_rate']:.0%}"
            )

    # Descriptive summary. The frozen grid is not a computed Pareto frontier.
    print("\n" + "=" * 60)
    print("CAPACITY-DETECTION OBSERVATIONS")
    print("=" * 60)
    print(f"{'Config':>12} {'BPT':>6} {'Probe AUC':>10}")
    print("-" * 45)
    for r in pareto_results:
        print(f"{r['label']:>12} {r['mean_net_bpt']:>6.2f} {r['probe_auc']:>10.3f}")

    # ===== PART 2: Probe Transfer =====
    print("\n" + "=" * 60)
    print("PART 2: PROBE TRANSFER")
    print("=" * 60)
    print("Train on raw stego -> test on each coupling config")
    print("(Can a defender use naive stego examples to catch coupled stego?)\n")

    # Use all raw snapshots for training
    raw_dataset = all_snapshots["raw"]

    transfer_results = []
    for params in COUPLING_GRID:
        label = params["label"]
        target_dataset = all_snapshots[label]
        metrics = train_and_evaluate(
            [raw_dataset.normal[i] for i in train_idx],
            [raw_dataset.stego[i] for i in train_idx],
            [target_dataset.normal[i] for i in test_idx],
            [target_dataset.stego[i] for i in test_idx],
            [sample_prompts[i] for i in test_idx],
        )

        transfer_results.append(
            {
                "train_condition": "raw",
                "test_condition": label,
                "probe_auc": metrics["auc"],
                "probe_auc_seed_ci_envelope": metrics["auc_seed_ci_envelope"],
                "probe_accuracy": metrics["accuracy"],
                "probe_seed_runs": metrics["seed_runs"],
            }
        )
        print(
            f"  raw -> {label:>12}: AUC {metrics['auc']:.3f}, Acc {metrics['accuracy']:.3f}"
        )

    # Also test reverse: train on aggressive, test on raw
    print("\nTrain on aggressive stego -> test on each config")
    agg_dataset = all_snapshots["aggressive"]

    reverse_transfer = []
    for params in COUPLING_GRID:
        label = params["label"]
        target_dataset = all_snapshots[label]
        metrics = train_and_evaluate(
            [agg_dataset.normal[i] for i in train_idx],
            [agg_dataset.stego[i] for i in train_idx],
            [target_dataset.normal[i] for i in test_idx],
            [target_dataset.stego[i] for i in test_idx],
            [sample_prompts[i] for i in test_idx],
        )

        reverse_transfer.append(
            {
                "train_condition": "aggressive",
                "test_condition": label,
                "probe_auc": metrics["auc"],
                "probe_auc_seed_ci_envelope": metrics["auc_seed_ci_envelope"],
                "probe_accuracy": metrics["accuracy"],
                "probe_seed_runs": metrics["seed_runs"],
            }
        )
        print(
            f"  aggressive -> {label:>12}: AUC {metrics['auc']:.3f}, Acc {metrics['accuracy']:.3f}"
        )

    # Transfer summary
    print("\n" + "=" * 60)
    print("TRANSFER MATRIX")
    print("=" * 60)
    print(f"{'':>12}", end="")
    for p in COUPLING_GRID:
        print(f" {p['label']:>10}", end="")
    print()

    print(f"{'raw ->':>12}", end="")
    for tr in transfer_results:
        print(f" {tr['probe_auc']:>10.3f}", end="")
    print()

    print(f"{'agg ->':>12}", end="")
    for tr in reverse_transfer:
        print(f" {tr['probe_auc']:>10.3f}", end="")
    print()

    # Descriptive checkpoint only. A transfer verdict requires a preregistered
    # decision threshold and a power statement; neither exists for this rerun.
    min_transfer = min(tr["probe_auc"] for tr in transfer_results)
    print(f"\nLowest observed raw->coupled AUC: {min_transfer:.3f}")
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
                "n_samples": n_samples,
                "n_capacity_trials": n_capacity_trials,
                "max_tokens": max_tokens,
                "probe_payload_bits": len(probe_secret.encode("ascii")) * 8,
                "probe_payload_retained": False,
                "key": _key_label(key),
                "length_match": "per-pair (normal matched to stego length)",
                "split": split_manifest(
                    sample_prompts, train_idx, test_idx, seed=split_seed
                ),
                "pareto": pareto_results,
                "transfer_raw_to_x": transfer_results,
                "transfer_agg_to_x": reverse_transfer,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
