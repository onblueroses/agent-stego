"""Evaluate the seeded multi-carrier hopping channel.

Offline and deterministic. Measures the properties the mechanism claims:

1. Seeded receiver delivery (exact recovery with the shared seed).
2. Naive exhaustive monitor blindness: rotation ensures no single chunk
   document is self-framing, so per-document exhaustive decoding catches
   nothing.
3. Schedule-search cost and ambiguity: an unscheduled monitor searches
   c^n schedules x n rotations; several candidates produce valid-looking
   frames (zero-filled), and only one carries the true payload. Brute force
   is demonstrated for small n and the search space reported for larger n.
4. Honest negatives: a single-carrier density detector is only partially
   effective (hopping dilutes per-carrier signal), while a full-registry
   scanner still catches every run.

Usage:
    python experiments/hopping_eval.py

Writes findings/hopping/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.channels.corpus import _hopping_covers, hopping_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.hopping import (
    HoppingChannel,
    brute_force_schedules,
    schedule,
    search_space,
)
from src.token_stego.text_channels import exhaustive_decode

FINDINGS_DIR = Path("findings/hopping")

MESSAGE = "HELLO"
NUM_BITS = len(MESSAGE.encode("ascii")) * 8  # 40


def run() -> None:
    channel = HoppingChannel()
    n_trials = 10

    # 1. Seeded receiver delivery + naive monitor blindness.
    delivered = 0
    naive_caught = 0
    for trial in range(n_trials):
        seed = 1_000 + trial
        n_chunks = channel.n_chunks(NUM_BITS)
        covers = _hopping_covers(schedule(seed, n_chunks))
        docs = channel.encode(MESSAGE, covers, seed)
        if channel.decode(docs, seed, NUM_BITS) == MESSAGE.encode("ascii"):
            delivered += 1
        if any(exhaustive_decode(doc) is not None for doc in docs):
            naive_caught += 1

    # 2. Schedule-search cost and ambiguity, demonstrated by brute force.
    space_report = {f"n={n}": search_space(n, NUM_BITS + 48) for n in (2, 3, 4, 6, 8)}
    brute_results: dict[str, Any] = {}
    for n_chunks in (2, 3):
        seed = 7
        # Choose a message length so the framed stream spans exactly n chunks:
        # total = 8*len + 48, n = ceil(total / 48).
        message = "H" if n_chunks == 2 else "HELLOHI"
        num_bits = len(message.encode("ascii")) * 8
        assert channel.n_chunks(num_bits) == n_chunks
        covers = _hopping_covers(schedule(seed, n_chunks))
        docs = channel.encode(message, covers, seed)
        candidates, attempts = brute_force_schedules(docs, num_bits)
        payloads = {candidate[2] for candidate in candidates}
        brute_results[f"n={n_chunks}"] = {
            "true_payload_recovered": message.encode("ascii") in payloads,
            "valid_looking_candidates": len(candidates),
            "attempts": attempts,
            "search_space": search_space(n_chunks, num_bits + 48),
            "ambiguous": len(candidates) > 1,
        }

    # 3. Honest negatives from the detector suite.
    from src.channels.corpus import clean_traces

    clean = clean_traces(24)
    stego = hopping_traces(12)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    invisible = evaluate_detector(
        by_name["calibrated:invisible_char_density"], stego, clean
    )
    diversity = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
    content_detectors = [
        d
        for d in detectors
        if d.feature
        in {
            "invisible_char_density",
            "nbsp_density",
            "homoglyph_density",
            "combining_mark_density",
            "non_ascii_density",
            "key_order_anomaly_fraction",
        }
    ]
    tp = sum(1 for run in stego if any(d.score(run) == 1.0 for d in content_detectors))
    fp = sum(1 for run in clean if any(d.score(run) == 1.0 for d in content_detectors))

    results = {
        "experiment": "seeded_multi_carrier_hopping",
        "mode": "offline-simulation",
        "message": MESSAGE,
        "chunk_size": channel.chunk_size,
        "carriers": list(channel.carriers),
        "seeded_receiver": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "naive_exhaustive_monitor": {
            "catch_rate": round(naive_caught / n_trials, 4),
            "note": "rotation ensures no chunk document is self-framing",
        },
        "schedule_search": {
            "space": space_report,
            "brute_force": brute_results,
            "note": "valid-looking frames are ambiguous; only one carries the true payload",
        },
        "honest_negatives": {
            "single_carrier_density_youden_j": invisible["youden_j"],
            "full_registry_catch_rate": round(tp / len(stego), 4),
            "full_registry_false_positive_rate": round(fp / len(clean), 4),
            "unique_tools_youden_j": diversity["youden_j"],
            "note": "hopping is a coordination mechanism, not invisibility",
        },
    }

    print("Seeded multi-carrier hopping (offline)")
    print(
        f"message={MESSAGE} chunk_size={channel.chunk_size} carriers={len(channel.carriers)}"
    )
    print(f"seeded receiver delivery: {delivered}/{n_trials}")
    print(f"naive exhaustive monitor catch: {naive_caught}/{n_trials}")
    for n, space in space_report.items():
        print(f"search space {n}: {space}")
    for n, entry in brute_results.items():
        print(
            f"brute force {n}: true={entry['true_payload_recovered']} "
            f"candidates={entry['valid_looking_candidates']} "
            f"attempts={entry['attempts']} ambiguous={entry['ambiguous']}"
        )
    print(
        f"honest negatives: single-carrier J={invisible['youden_j']}, "
        f"full-registry catch {tp}/{len(stego)}, FPR {fp}/{len(clean)}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
