"""Evaluate the resource-usage side channel (CPU-time carrier).

Offline and deterministic. Measures the claims from
analysis/resource_side_channel.md:

1. Round trip: bits ride in per-call CPU consumption with wide margins.
2. Robustness vs compute jitter (cliff at half the margin).
3. Content-guard blindness: every content/structural scanner is blind (the
   compute pattern bypasses content inspection).
4. Honest negatives: a resource-aware monitor catches it at J = 1.0, and
   the framing overhead forces many calls (volume tell).

Usage:
    python experiments/resource_eval.py

Writes findings/resource/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import clean_traces, resource_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.resource_side import ResourceExecutor, ResourceSideChannel

FINDINGS_DIR = Path("findings/resource")
MESSAGE = "HI"


def run() -> None:
    channel = ResourceSideChannel()
    n_trials = 6
    delivered = 0
    for i in range(n_trials):
        executor = ResourceExecutor(jitter_ms=20.0, seed=200 + i)
        traces = executor.run(channel.encode(MESSAGE))
        if channel.decode(traces) == MESSAGE:
            delivered += 1

    robustness: dict[str, float] = {}
    for jitter in (0, 50, 100, 180, 200):
        ok = 0
        trials = 4
        for seed in range(trials):
            executor = ResourceExecutor(jitter_ms=jitter, seed=seed)
            traces = executor.run(channel.encode(MESSAGE))
            ok += channel.decode(traces) == MESSAGE
        robustness[str(jitter)] = round(ok / trials, 3)

    clean = clean_traces(24)
    stego = resource_traces(8)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    content_features = [
        "invisible_char_density",
        "nbsp_density",
        "homoglyph_density",
        "combining_mark_density",
        "non_ascii_density",
        "key_order_anomaly_fraction",
        "duplicate_key_fraction",
        "synonym_surface_fraction",
        "number_format_anomaly_fraction",
    ]
    content_blind = {
        feature: evaluate_detector(by_name[f"calibrated:{feature}"], stego, clean)[
            "youden_j"
        ]
        for feature in content_features
    }
    cpu = evaluate_detector(by_name["calibrated:cpu_ms_std"], stego, clean)
    call_count = evaluate_detector(by_name["calibrated:call_count"], stego, clean)

    results = {
        "experiment": "resource_usage_side_channel",
        "mode": "offline-simulation",
        "message": MESSAGE,
        "short_ms": channel.short_ms,
        "long_ms": channel.long_ms,
        "round_trip": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "robustness_by_jitter_ms": robustness,
        "content_guard_blindness_youden_j": content_blind,
        "detection": {
            "cpu_ms_std_youden_j": cpu["youden_j"],
            "call_count_youden_j": call_count["youden_j"],
        },
    }

    print("Resource-usage side channel (offline)")
    print(f"message={MESSAGE} short={channel.short_ms}ms long={channel.long_ms}ms")
    print(f"round-trip delivery: {delivered}/{n_trials}")
    print("robustness (delivery vs jitter ms):", robustness)
    print("content-guard blindness (all J):", content_blind)
    print(
        f"detection: cpu-std J={cpu['youden_j']}, call-count J={call_count['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
