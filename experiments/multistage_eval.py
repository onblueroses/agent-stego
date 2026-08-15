"""Evaluate multi-stage cross-surface composition.

Offline and deterministic. Measures the claims from
analysis/multistage_composition.md:

1. Joint decode: the receiver with the stage map recovers the payload exactly.
2. Per-surface blindness: no single surface carries a self-framing fragment,
   so a monitor restricted to one surface (tool-choice, file content, or
   key order) recovers nothing.
3. Coordination cost: the stage map is the key — wrong stage order fails
   closed, and an unscheduled monitor must search the assignment space.
4. Honest negatives: each stage leaves its surface's statistical signature,
   so the full registry (and call count alone) catches the family.

Usage:
    python experiments/multistage_eval.py

Writes findings/multistage/results_<timestamp>.json
"""

import itertools
import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import clean_traces, multistage_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.multistage import (
    STAGES,
    MultiStageChannel,
    MultiStageExecutor,
)
from src.token_stego.text_channels import _unframe_bits

FINDINGS_DIR = Path("findings/multistage")
MESSAGE = "HELLO"


def run() -> None:
    n_trials = 6
    delivered = 0
    surface_blind = {stage: 0 for stage in STAGES}

    runs = []
    for i in range(n_trials):
        channel = MultiStageChannel(report_cover=("report " * 60) + f"trial {i}")
        executor = MultiStageExecutor(start_time=9_500_000.0 + i * 50.0)
        traces = executor.run(channel.encode(MESSAGE))
        runs.append((traces, channel))
        if channel.decode(traces) == MESSAGE:
            delivered += 1
        for stage in STAGES:
            if channel.decode_surface(traces, stage) == "":
                surface_blind[stage] += 1

    # Coordination cost: wrong stage orders fail closed.
    traces, channel = runs[0][0], runs[0][1]
    bits = channel.surface_bits(traces)
    orders = list(itertools.permutations(STAGES))
    wrong_orders_fail = sum(
        1
        for order in orders
        if order != STAGES and _unframe_bits("".join(bits[s] for s in order)) is None
    )

    # Honest negatives from the detector suite.
    clean = clean_traces(24)
    stego = multistage_traces(8)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    call_count = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
    invisible = evaluate_detector(
        by_name["calibrated:invisible_char_density"], stego, clean
    )

    results = {
        "experiment": "multi_stage_cross_surface",
        "mode": "offline-simulation",
        "message": MESSAGE,
        "stages": list(STAGES),
        "joint_decode": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "per_surface_monitor_blind": {
            stage: round(blind / n_trials, 4) for stage, blind in surface_blind.items()
        },
        "coordination_cost": {
            "stage_permutations": len(orders),
            "wrong_orders_failing_closed": wrong_orders_fail,
            "note": "the stage map is the key; wrong assignment fails the magic check",
        },
        "honest_negatives": {
            "call_count_youden_j": call_count["youden_j"],
            "invisible_char_youden_j": invisible["youden_j"],
            "note": "every stage leaves its surface's signature; multi-stage is "
            "coordination, not invisibility",
        },
    }

    print("Multi-stage cross-surface composition (offline)")
    print(f"message={MESSAGE} stages={list(STAGES)}")
    print(f"joint decode delivery: {delivered}/{n_trials}")
    for stage, blind in surface_blind.items():
        print(f"  per-surface monitor blind ({stage}): {blind}/{n_trials}")
    print(f"wrong stage orders failing closed: {wrong_orders_fail}/{len(orders) - 1}")
    print(
        f"honest negatives: call-count J={call_count['youden_j']}, "
        f"invisible-char J={invisible['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
