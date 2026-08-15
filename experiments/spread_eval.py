"""Evaluate the spread-spectrum multi-carrier channel.

Offline and deterministic. Measures the claims from
analysis/spread_spectrum.md:

1. Round trip: the framed payload is replicated across several carriers.
2. Strip-one robustness: a monitor canonicalizing exactly one carrier class
   (NFC, whitespace normalization, invisible-char stripping) cannot kill the
   channel; only stripping every carrier does.
3. Honest tradeoff: redundancy multiplies the statistical signal — several
   content scanners fire at J = 1.0.

Usage:
    python experiments/spread_eval.py

Writes findings/spread/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import (
    _LETTER_RICH_COVERS,
    _PLAIN_COVERS,
    _SPACE_RICH_COVERS,
    clean_traces,
    spread_traces,
)
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.spread import SpreadChannel, strip_carrier

FINDINGS_DIR = Path("findings/spread")
MESSAGE = "HELLO"


def run() -> None:
    channel = SpreadChannel()
    covers = [_PLAIN_COVERS[0], _SPACE_RICH_COVERS[0], _LETTER_RICH_COVERS[0]]
    n_trials = 6

    delivered = 0
    strip_one_ok = {carrier: 0 for carrier in channel.carriers}
    strip_all_dead = 0
    for i in range(n_trials):
        docs = channel.encode(MESSAGE, covers)
        if channel.decode(docs) == MESSAGE.encode("ascii"):
            delivered += 1
        for carrier in channel.carriers:
            if channel.decode_after_strip(docs, carrier) == MESSAGE.encode("ascii"):
                strip_one_ok[carrier] += 1
        all_stripped = [
            strip_carrier(carrier, doc) for carrier, doc in zip(channel.carriers, docs)
        ]
        if channel.decode(all_stripped) is None:
            strip_all_dead += 1

    clean = clean_traces(24)
    stego = spread_traces(8)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    invisible = evaluate_detector(
        by_name["calibrated:invisible_char_density"], stego, clean
    )
    nbsp = evaluate_detector(by_name["calibrated:nbsp_density"], stego, clean)
    homoglyph = evaluate_detector(by_name["calibrated:homoglyph_density"], stego, clean)

    results = {
        "experiment": "spread_spectrum_multicarrier",
        "mode": "offline-simulation",
        "message": MESSAGE,
        "carriers": list(channel.carriers),
        "round_trip": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "strip_one_still_delivers": {
            carrier: round(ok / n_trials, 4) for carrier, ok in strip_one_ok.items()
        },
        "strip_all_dead": round(strip_all_dead / n_trials, 4),
        "honest_negative_detection": {
            "invisible_char_youden_j": invisible["youden_j"],
            "nbsp_density_youden_j": nbsp["youden_j"],
            "homoglyph_density_youden_j": homoglyph["youden_j"],
            "note": "redundancy multiplies the statistical signal",
        },
    }

    print("Spread-spectrum multi-carrier channel (offline)")
    print(f"message={MESSAGE} carriers={list(channel.carriers)}")
    print(f"round-trip delivery: {delivered}/{n_trials}")
    for carrier, ok in strip_one_ok.items():
        print(f"  strip {carrier:10s} still delivers: {ok}/{n_trials}")
    print(f"strip all dead: {strip_all_dead}/{n_trials}")
    print(
        f"honest negative: invisible J={invisible['youden_j']}, "
        f"nbsp J={nbsp['youden_j']}, homoglyph J={homoglyph['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
