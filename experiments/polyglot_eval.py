"""Evaluate polyglot documents (JSON duplicate-key parser differential).

Offline and deterministic. Measures the claims from
analysis/polyglot_documents.md:

1. Round trip: the receiver (byte-preserving, duplicate-aware) recovers the
   payload; the blind monitor (first-wins / canonicalizing) sees a normal
   config.
2. Parser differential: any parse-and-reserialize path (first-wins or
   last-wins) collapses duplicates and destroys the channel — the axis is
   canonicalization, not which duplicate value wins.
3. Honest negative: a duplicate-aware scanner catches the family at J = 1.0;
   the mechanism protects against canonicalizing monitors, not against
   scanners that look for the anomaly.

Usage:
    python experiments/polyglot_eval.py

Writes findings/polyglot/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import _polyglot_cover, clean_traces, polyglot_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.polyglot import DuplicateKeyChannel, first_wins

FINDINGS_DIR = Path("findings/polyglot")
PAYLOAD = b"HELLO"


def run() -> None:
    channel = DuplicateKeyChannel()
    n_trials = 6

    delivered = 0
    blind_first_wins = 0
    collapsed = 0
    for i in range(n_trials):
        stego = channel.encode(PAYLOAD, _polyglot_cover(i))
        if channel.decode(stego) == PAYLOAD:
            delivered += 1
        if channel.decode(first_wins(stego)) is None:
            blind_first_wins += 1
        last_wins_text = json.dumps(json.loads(stego), indent=2)
        if channel.decode(last_wins_text) is None:
            collapsed += 1

    # Detector separation (honest negative).
    clean = clean_traces(24)
    stego_traces = polyglot_traces(12)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    duplicates = evaluate_detector(
        by_name["calibrated:duplicate_key_fraction"], stego_traces, clean
    )
    diversity = evaluate_detector(
        by_name["calibrated:unique_tools"], stego_traces, clean
    )

    results = {
        "experiment": "polyglot_duplicate_keys",
        "mode": "offline-simulation",
        "message": PAYLOAD.decode(),
        "capacity_keys": channel.capacity(_polyglot_cover(0)),
        "round_trip": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "parser_differential": {
            "first_wins_blind_rate": round(blind_first_wins / n_trials, 4),
            "reserialized_collapsed_rate": round(collapsed / n_trials, 4),
            "note": "any parse-and-reserialize path collapses duplicates; the axis "
            "is canonicalization, not which duplicate value wins",
        },
        "honest_negative_detection": {
            "duplicate_key_fraction_youden_j": duplicates["youden_j"],
            "unique_tools_youden_j": diversity["youden_j"],
        },
    }

    print("Polyglot documents: duplicate-key parser differential (offline)")
    print(f"message={PAYLOAD.decode()} capacity={results['capacity_keys']} keys")
    print(f"round-trip delivery: {delivered}/{n_trials}")
    print(f"first-wins monitor blind: {blind_first_wins}/{n_trials}")
    print(f"re-serialized (any win-order) collapsed: {collapsed}/{n_trials}")
    print(
        f"honest negative: duplicate-scanner J={duplicates['youden_j']}, "
        f"diversity J={diversity['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
