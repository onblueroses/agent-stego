"""Evaluate the trace detector registry against the synthetic corpus.

Offline and deterministic. Builds the labelled corpus, calibrates detector
thresholds on clean traces ONLY (mean + 5*std), then measures AUC and Youden's
J of every detector against every channel family. The output matrix shows
which detectors catch which channels and — equally important — which
detectors do not.

Usage:
    python experiments/trace_detection_eval.py

Writes findings/trace_detection/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import corpus
from src.channels.detector import build_detector_registry, evaluate_registry

FINDINGS_DIR = Path("findings/trace_detection")

DETECTOR_LABELS = {
    "calibrated:invisible_char_density": "invisible-char density",
    "calibrated:nbsp_density": "NBSP density",
    "calibrated:homoglyph_density": "homoglyph density",
    "calibrated:combining_mark_density": "combining-mark density",
    "calibrated:non_ascii_density": "non-ASCII density",
    "calibrated:key_order_anomaly_fraction": "key-order anomaly",
    "calibrated:single_tool_fraction": "single-tool fraction",
    "calibrated:call_count": "call count",
    "calibrated:unique_tools": "unique tools",
    "calibrated:mean_latency": "mean latency",
}


def run() -> None:
    data = corpus()
    clean = data.pop("clean")

    detectors = build_detector_registry(clean, k=5.0)

    results: dict[str, object] = {
        "experiment": "trace_detector_registry",
        "mode": "offline-simulation",
        "calibration": "clean-only, mean + 5*std",
        "n_clean": len(clean),
        "n_stego_per_family": max(len(v) for v in data.values()),
        "matrix": {},
    }

    print("Trace detector registry vs synthetic corpus (offline)")
    print(f"clean={len(clean)}  stego per family={max(len(v) for v in data.values())}")
    print(f"{'detector':<28s} {'family':<28s} {'AUC':>6s} {'J':>6s}")
    print("-" * 74)

    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for family, stego in sorted(data.items()):
        evaluated = evaluate_registry(detectors, stego, clean)
        matrix[family] = evaluated
        for detector in detectors:
            short = DETECTOR_LABELS.get(detector.name, detector.name)
            stats = evaluated[detector.name]
            print(
                f"{short:<28s} {family:<28s} {stats['auc']:6.2f} {stats['youden_j']:6.2f}"
            )

    results["matrix"] = matrix
    # Per-family best detector (highest J) and per-detector false-positive rate.
    results["family_best"] = {
        family: max(
            ((name, stats["youden_j"]) for name, stats in evals.items()),
            key=lambda pair: pair[1],
        )
        for family, evals in matrix.items()
    }

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
