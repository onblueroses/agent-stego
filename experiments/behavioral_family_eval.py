"""Behavioral side-channel family comparison: timing vs resource vs error.

All three are content-independent (content guards blind) and differ in how
*observable* they are to standard monitoring:

- timing: needs latency instrumentation
- resource: needs compute instrumentation
- error: visible to any reliability/observability tooling (failures are
  structurally loud)

Each is caught at J = 1.0 by its modality-aware detector; all share the
framing-overhead volume tell.

Usage:
    python experiments/behavioral_family_eval.py

Writes findings/behavioral_family/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import (
    clean_traces,
    error_traces,
    length_traces,
    resource_traces,
    timing_traces,
)
from src.channels.detector import build_detector_registry, evaluate_detector

FINDINGS_DIR = Path("findings/behavioral_family")

FAMILIES = {
    "timing": (timing_traces, "latency_std"),
    "resource": (resource_traces, "cpu_ms_std"),
    "length": (length_traces, "token_count_std"),
    "error": (error_traces, "error_fraction"),
}

CONTENT_FEATURES = [
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


def run() -> None:
    clean = clean_traces(24)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}

    rows: dict[str, dict[str, float]] = {}
    for name, (builder, monitor) in FAMILIES.items():
        stego = builder(8)
        content_blind = sum(
            1
            for feature in CONTENT_FEATURES
            if evaluate_detector(by_name[f"calibrated:{feature}"], stego, clean)[
                "youden_j"
            ]
            == 0.0
        )
        modality = evaluate_detector(by_name[f"calibrated:{monitor}"], stego, clean)
        volume = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        rows[name] = {
            "content_scanners_blind": content_blind,
            "content_scanners_total": len(CONTENT_FEATURES),
            "modality_monitor_youden_j": modality["youden_j"],
            "call_count_youden_j": volume["youden_j"],
        }

    results = {
        "experiment": "behavioral_side_channel_family",
        "mode": "offline-simulation",
        "families": rows,
        "stealth_ordering": {
            "timing ~ resource > length > error": (
                "timing and resource need modality-specific instrumentation; "
                "length uses standard usage telemetry (completion tokens); "
                "error patterns are visible to any reliability tooling"
            )
        },
    }

    print("Behavioral side-channel family (offline)")
    for name, row in rows.items():
        print(
            f"  {name:8s} content-blind {row['content_scanners_blind']}/"
            f"{row['content_scanners_total']}  modality J={row['modality_monitor_youden_j']}  "
            f"volume J={row['call_count_youden_j']}"
        )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
