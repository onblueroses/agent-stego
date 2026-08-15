"""Evaluate the number-format (state-value) carrier.

Offline and deterministic. Measures the claims from
analysis/number_format_carrier.md:

1. Round trip: the byte-preserving receiver recovers the payload from
   value-equivalent numeric spellings.
2. Canonicalization blindness: a monitor that parses and re-serializes
   (normalizing integral floats) collapses every variant and is blind.
3. Honest negative: the spelling scanner separates the family at J = 1.0
   while character, structural, and diversity scanners are blind (isolation).

Usage:
    python experiments/numberfmt_eval.py

Writes findings/numberfmt/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import _number_cover, clean_traces, numberfmt_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.numberfmt import NumberFormatChannel, canonicalize

FINDINGS_DIR = Path("findings/numberfmt")
PAYLOAD = b"HELLO"


def run() -> None:
    channel = NumberFormatChannel()
    cover = _number_cover(0)
    n_trials = 6

    delivered = 0
    canonical_blind = 0
    for i in range(n_trials):
        stego = channel.encode(PAYLOAD, _number_cover(i))
        if channel.decode(stego) == PAYLOAD:
            delivered += 1
        if channel.decode(canonicalize(stego)) is None:
            canonical_blind += 1

    clean = clean_traces(24)
    stego_traces = numberfmt_traces(12)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    numbers = evaluate_detector(
        by_name["calibrated:number_format_anomaly_fraction"], stego_traces, clean
    )
    invisible = evaluate_detector(
        by_name["calibrated:invisible_char_density"], stego_traces, clean
    )
    key_order = evaluate_detector(
        by_name["calibrated:key_order_anomaly_fraction"], stego_traces, clean
    )
    diversity = evaluate_detector(
        by_name["calibrated:unique_tools"], stego_traces, clean
    )

    results = {
        "experiment": "number_format_carrier",
        "mode": "offline-simulation",
        "message": PAYLOAD.decode(),
        "capacity_numbers": channel.capacity(cover),
        "round_trip": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "blindness": {
            "canonicalizing_monitor_blind": round(canonical_blind / n_trials, 4),
            "note": "the axis is canonicalization: parse + re-serialize collapses "
            "1000/1e3, 1.5/1.50, 0/-0",
        },
        "honest_negative_detection": {
            "number_format_youden_j": numbers["youden_j"],
            "invisible_char_youden_j": invisible["youden_j"],
            "key_order_youden_j": key_order["youden_j"],
            "unique_tools_youden_j": diversity["youden_j"],
            "note": "isolated: only the spelling scanner sees it",
        },
    }

    print("Number-format (state-value) carrier (offline)")
    print(f"message={PAYLOAD.decode()} capacity={results['capacity_numbers']} numbers")
    print(f"round-trip delivery: {delivered}/{n_trials}")
    print(f"canonicalizing monitor blind: {canonical_blind}/{n_trials}")
    print(
        f"honest negative: number-format J={numbers['youden_j']}, "
        f"invisible J={invisible['youden_j']}, key-order J={key_order['youden_j']}, "
        f"diversity J={diversity['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
