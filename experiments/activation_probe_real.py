"""Real activation-probe detection with the group-disjoint protocol.

The round-1 detection-duality table historically reported activation-probe
AUC ~1.0, later invalidated by train/test prompt-group leakage. This
experiment re-measures it correctly on real small-model data:

- stego text via ArithmeticStegoChannel (the current coder);
- normal text via MATCHED sampling (temperature 1.0, top-p 1.0) — the
  output-level baseline that surprise detection could not separate;
- hidden states captured per text (last-position, all captured layers);
- a 2-layer MLP probe trained on a GROUP-DISJOINT split of prompt groups
  (experiment_design.grouped_train_test_indices), three seeds, bootstrap CI.

Output-level detection was shown surprise-preserving (round 41); the open
question is whether activations still leak. Either outcome is the measured
completion of the duality.

Usage:
    python experiments/activation_probe_real.py

Writes findings/activation_probe_real/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

OUT = Path("findings/activation_probe_real")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_PROMPTS = 12
NORMAL_TOKENS = 40


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    _log(f"activation_probe_real start; model={MODEL} prompts={N_PROMPTS}")
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
    channel = ArithmeticStegoChannel(model)

    prompts = PROBE_PROMPTS[:N_PROMPTS]
    normal_snaps: list = []
    stego_snaps: list = []
    groups: list[str] = []

    torch.manual_seed(0)
    for i, prompt in enumerate(prompts):
        # Stego text.
        stego = channel.encode("HELLO", prompt, max_tokens=250)

        # Matched normal: temperature 1.0, top-p 1.0 sampling.
        current = list(model.tokenize(prompt))
        normal_ids: list[int] = []
        for _ in range(NORMAL_TOKENS):
            dist = model.get_distribution(current)
            probs = torch.tensor(dist)
            token_id = int(torch.multinomial(probs, 1).item())
            if token_id == model.tokenizer.eos_token_id:
                break
            normal_ids.append(token_id)
            current.append(token_id)
        normal_text = model.detokenize(normal_ids)

        # Capture hidden states: one forward pass over prompt + text.
        with LatentCapture(model) as cap:
            model.get_distribution(list(model.tokenize(prompt + stego.text)))
            stego_snap = cap.snapshot()
        with LatentCapture(model) as cap:
            model.get_distribution(list(model.tokenize(prompt + normal_text)))
            normal_snap = cap.snapshot()

        normal_snaps.append(normal_snap)
        stego_snaps.append(stego_snap)
        groups.append(prompt)
        _log(
            f"  prompt {i}: captured (stego {len(stego.text)} chars, normal {len(normal_text)} chars)"
        )

    # Group-disjoint split: whole prompt groups, never repeated observations.
    train_idx, test_idx = grouped_train_test_indices(
        groups, seed=42, train_fraction=0.7
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
        _log(
            f"  probe seed {seed}: test AUC={result['auc']:.3f} "
            f"CI={[round(x, 3) for x in result['auc_ci_95']]} (n_test={len(test_idx)} groups)"
        )

    mean_auc = sum(aucs) / len(aucs)

    results = {
        "experiment": "activation_probe_real_group_disjoint",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": MODEL,
        "n_prompt_groups": N_PROMPTS,
        "split": "group-disjoint (prompt groups never cross train/test)",
        "n_train_groups": len(train_idx),
        "n_test_groups": len(test_idx),
        "normal_baseline": "matched sampling T=1.0 top_p=1.0",
        "probe_auc_mean": round(mean_auc, 4),
        "probe_seeds": details,
    }
    (OUT / f"results_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    _log(f"activation_probe_real done; mean AUC={mean_auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
