"""Evaluate the offline-simulated semantic (synonym-choice) carrier.

Deterministic, no model required — this isolates the word-choice mechanism
that the overnight program's P2 phase showed small models cannot execute, so
its detection profile is measurable without a capable sender.

Measured claims:
1. Round trip: the word-preserving receiver recovers the payload.
2. Lexical-normalization blindness: a canonicalizing (lemmatizing) monitor
   collapses synonyms and is blind; the character-channel exhaustive decoder
   is also blind (distinct taxonomy axis).
3. Honest negative: a synonym-surface scanner separates the family at J=1.0
   while character and diversity scanners are blind (isolation).

Usage:
    python experiments/semantic_eval.py

Writes findings/semantic/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import clean_traces, semantic_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.semantic import SynonymChoiceChannel, _anchor_cover, canonicalize
from src.token_stego.text_channels import exhaustive_decode

FINDINGS_DIR = Path("findings/semantic")
PAYLOAD = b"HELLO"


def run() -> None:
    channel = SynonymChoiceChannel()
    cover = _anchor_cover(120)
    n_trials = 6

    delivered = 0
    canonical_blind = 0
    exhaustive_blind = 0
    for i in range(n_trials):
        stego = channel.encode(PAYLOAD, cover + f" This is report number {i}.")
        if channel.decode(stego) == PAYLOAD:
            delivered += 1
        if channel.decode(canonicalize(stego)) is None:
            canonical_blind += 1
        if exhaustive_decode(stego) is None:
            exhaustive_blind += 1

    clean = clean_traces(24)
    stego_traces = semantic_traces(12)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    synonym = evaluate_detector(
        by_name["calibrated:synonym_surface_fraction"], stego_traces, clean
    )
    invisible = evaluate_detector(
        by_name["calibrated:invisible_char_density"], stego_traces, clean
    )
    diversity = evaluate_detector(
        by_name["calibrated:unique_tools"], stego_traces, clean
    )

    results = {
        "experiment": "semantic_synonym_choice",
        "mode": "offline-simulation",
        "message": PAYLOAD.decode(),
        "capacity_anchors": channel.capacity(cover) // 2,
        "round_trip": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "blindness": {
            "canonicalizing_monitor_blind": round(canonical_blind / n_trials, 4),
            "character_exhaustive_decoder_blind": round(exhaustive_blind / n_trials, 4),
            "note": "blindness axis is lexical normalization; the semantic carrier "
            "is a distinct taxonomy axis from character and structural",
        },
        "honest_negative_detection": {
            "synonym_surface_youden_j": synonym["youden_j"],
            "invisible_char_youden_j": invisible["youden_j"],
            "unique_tools_youden_j": diversity["youden_j"],
            "note": "isolated: only the word-surface scanner sees it",
        },
    }

    print("Semantic synonym-choice carrier (offline)")
    print(f"message={PAYLOAD.decode()} capacity={results['capacity_anchors']} anchors")
    print(f"round-trip delivery: {delivered}/{n_trials}")
    print(f"canonicalizing monitor blind: {canonical_blind}/{n_trials}")
    print(f"character exhaustive decoder blind: {exhaustive_blind}/{n_trials}")
    print(
        f"honest negative: synonym J={synonym['youden_j']}, "
        f"invisible J={invisible['youden_j']}, diversity J={diversity['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
