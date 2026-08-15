"""Activation-probe evasion: does distribution coupling reduce the signal?

Round 46 proved the uncoupled coder is activation-trivial (group-disjoint
probe AUC 1.000). The historical "distribution coupling" evasion was
invalidated with the same protocol errors. This experiment measures coupling
against the REAL probe:

- coupled stego: ArithmeticStegoChannel with temperature/top-p applied
  before coding (moderate 0.9/0.9, aggressive 0.7/0.85);
- matched normal baseline: sampled with the SAME temperature/top-p (a fair
  defender baseline — coupling is part of the channel);
- group-disjoint probe per config, three seeds, bootstrap CIs.

Coupling forces the coder to pick from the tokens normal sampling uses, so
the activation signal may drop — the evasion question, measured.

Usage:
    python experiments/activation_probe_coupled.py

Writes findings/activation_probe_coupled/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

OUT = Path("findings/activation_probe_coupled")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 12
NORMAL_TOKENS = 40
CONFIGS = [
    {"name": "moderate", "temperature": 0.9, "top_p": 0.9},
    {"name": "aggressive", "temperature": 0.7, "top_p": 0.85},
]


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    _log(f"activation_probe_coupled start; model={MODEL}")
    from src.token_stego.activation_probe import evaluate_probe, train_probe
    from src.token_stego.channel import ArithmeticStegoChannel
    from src.token_stego.experiment_design import (
        PROBE_PROMPTS,
        grouped_train_test_indices,
    )
    from src.token_stego.latent import LatentCapture
    from src.token_stego.model import StegoModel

    model = StegoModel(MODEL)
    model.load()
    prompts = PROBE_PROMPTS[:N_PROMPTS]

    config_results: dict[str, object] = {}
    for config in CONFIGS:
        temp = config["temperature"]
        top_p = config["top_p"]
        channel = ArithmeticStegoChannel(model, temperature=temp, top_p=top_p)
        normal_snaps: list = []
        stego_snaps: list = []
        used_prompts: list[str] = []
        failures = 0

        torch.manual_seed(0)
        for i, prompt in enumerate(prompts):
            # Coupled stego (may fail to complete if capacity is insufficient).
            try:
                stego = channel.encode("HELLO", prompt, max_tokens=400)
            except Exception as exc:
                failures += 1
                _log(f"  {config['name']} prompt {i}: encode failed ({exc}); skipping")
                continue

            # Matched normal: same temperature/top-p sampling.
            current = list(model.tokenize(prompt))
            normal_ids: list[int] = []
            for _ in range(NORMAL_TOKENS):
                dist = model.get_distribution(current, temperature=temp, top_p=top_p)
                probs = torch.tensor(dist)
                token_id = int(torch.multinomial(probs, 1).item())
                if token_id == model.tokenizer.eos_token_id:
                    break
                normal_ids.append(token_id)
                current.append(token_id)
            normal_text = model.detokenize(normal_ids)

            with LatentCapture(model) as cap:
                model.get_distribution(list(model.tokenize(prompt + stego.text)))
                stego_snaps.append(cap.snapshot())
            with LatentCapture(model) as cap:
                model.get_distribution(list(model.tokenize(prompt + normal_text)))
                normal_snaps.append(cap.snapshot())
            used_prompts.append(prompt)
            _log(f"  {config['name']} prompt {i}: captured")

        if failures:
            _log(f"  {config['name']}: {failures}/{N_PROMPTS} encode failures")

        train_idx, test_idx = grouped_train_test_indices(
            used_prompts, seed=42, train_fraction=0.7
        )
        test_snaps = [stego_snaps[i] for i in test_idx] + [
            normal_snaps[i] for i in test_idx
        ]
        test_labels = [1] * len(test_idx) + [0] * len(test_idx)

        aucs: list[float] = []
        details: list[dict] = []
        for seed in (0, 1, 2):
            probe = train_probe(
                [normal_snaps[i] for i in train_idx],
                [stego_snaps[i] for i in train_idx],
                epochs=120,
                lr=1e-3,
                seed=seed,
            )
            result = evaluate_probe(probe, test_snaps, test_labels, seed=seed)
            aucs.append(result["auc"])
            details.append(
                {"seed": seed, "auc": result["auc"], "auc_ci_95": result["auc_ci_95"]}
            )
        mean_auc = sum(aucs) / len(aucs)
        config_results[config["name"]] = {
            "temperature": temp,
            "top_p": top_p,
            "encode_failures": failures,
            "probe_auc_mean": round(mean_auc, 4),
            "probe_seeds": details,
            "n_test_groups": len(test_idx),
        }
        _log(f"  {config['name']}: mean test AUC={mean_auc:.3f}")

    results = {
        "experiment": "activation_probe_coupled",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "uncoupled_reference_auc": 1.0,  # round 46
        "normal_baseline": "matched temperature/top-p per config",
        "configs": config_results,
    }
    (OUT / f"results_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    _log("activation_probe_coupled done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
