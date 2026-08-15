"""Evaluate the carrier meta-channel (keyless cross-surface hopping).

Offline and deterministic. Measures the claims from
analysis/meta_channel.md:

1. Joint decode without any shared key: the schedule travels in the
   structural chunk, so the receiver needs no seed or side information.
2. Per-surface monitor blindness: text fragments are unframed (no payload);
   the structural chunk yields the schedule but no payload.
3. Honest negatives: every chunk leaves its surface's signature (the full
   registry catches the family), and the header is a single point of
   failure (canonicalizing the structural surface kills the whole channel).

Usage:
    python experiments/meta_hop_eval.py

Writes findings/meta_hop/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import (
    _ACCENTED_COVERS,
    _LETTER_RICH_COVERS,
    _PLAIN_COVERS,
    _SPACE_RICH_COVERS,
    _cover_config,
    clean_traces,
    meta_hop_traces,
)
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.meta_hop import MetaHopChannel
from src.token_stego.text_channels import exhaustive_decode

FINDINGS_DIR = Path("findings/meta_hop")
PAYLOAD = "HELLO"


def _cover(carrier: str, i: int) -> str:
    pools = {
        "zero_width": _PLAIN_COVERS,
        "variation_selector": _PLAIN_COVERS,
        "tag_char": _PLAIN_COVERS,
        "nbsp": _SPACE_RICH_COVERS,
        "nfd": _ACCENTED_COVERS,
        "homoglyph": _LETTER_RICH_COVERS,
    }
    return pools[carrier][i % len(pools[carrier])]


def run() -> None:
    channel = MetaHopChannel()
    schedules = [
        ["zero_width", "nbsp"],
        ["homoglyph", "nfd"],
        ["tag_char", "zero_width"],
    ]
    n_trials = len(schedules)

    delivered = 0
    text_true_recovery = 0
    text_ambiguous_frames = 0
    header_ok = 0
    for i, carriers in enumerate(schedules):
        docs = channel.encode(
            PAYLOAD,
            _cover_config(i),
            [_cover(c, i) for c in carriers],
            carriers,
        )
        if channel.decode(docs) == PAYLOAD.encode("ascii"):
            delivered += 1
        for doc in docs[1:]:
            found = exhaustive_decode(doc)
            if found == PAYLOAD.encode("ascii"):
                text_true_recovery += 1
            elif found is not None:
                text_ambiguous_frames += 1
        header = channel.extract_header(docs)
        if header is not None and channel.decode([docs[0]]) is None:
            header_ok += 1

    # Single point of failure: canonicalizing the structural surface.
    from src.channels.json_order import canonicalize

    docs = channel.encode(
        PAYLOAD,
        _cover_config(0),
        [_cover("zero_width", 0), _cover("nbsp", 0)],
        ["zero_width", "nbsp"],
    )
    docs[0] = canonicalize(docs[0])
    header_killed = channel.decode(docs) is None

    clean = clean_traces(24)
    stego_traces = meta_hop_traces(8)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    key_order = evaluate_detector(
        by_name["calibrated:key_order_anomaly_fraction"], stego_traces, clean
    )

    results = {
        "experiment": "carrier_meta_channel",
        "mode": "offline-simulation",
        "message": PAYLOAD,
        "schedules": schedules,
        "joint_decode": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "per_surface_monitor": {
            "text_surface_true_payload_recovery": round(
                text_true_recovery / (2 * n_trials), 4
            ),
            "text_surface_ambiguous_garbage_frames": round(
                text_ambiguous_frames / (2 * n_trials), 4
            ),
            "structural_surface_schedule_only": round(header_ok / n_trials, 4),
            "note": "the text surface never recovers the true payload; chunk 1 on "
            "substitution-based carriers (homoglyph/nbsp/nfd) leaks an ambiguous "
            "zero-filled frame (the round-4 ambiguity, localized); insertion "
            "carriers are fully blind",
        },
        "keyless": {"note": "no shared seed; the schedule travels in chunk 0"},
        "single_point_of_failure": {
            "canonicalized_header_kills_channel": header_killed,
            "note": "canonicalizing the structural surface destroys the schedule "
            "and therefore the whole channel",
        },
        "honest_negative_detection": {
            "key_order_anomaly_youden_j": key_order["youden_j"],
            "note": "every chunk leaves its surface's signature",
        },
    }

    print("Carrier meta-channel (offline)")
    print(f"message={PAYLOAD} schedules={len(schedules)}")
    print(f"joint decode delivery: {delivered}/{n_trials} (keyless)")
    print(f"text-surface true payload recovery: {text_true_recovery}/{2 * n_trials}")
    print(
        f"text-surface ambiguous garbage frames: {text_ambiguous_frames}/{2 * n_trials}"
    )
    print(f"structural-surface monitor gets schedule only: {header_ok}/{n_trials}")
    print(f"canonicalized header kills channel: {header_killed}")
    print(f"honest negative: key-order J={key_order['youden_j']}")

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
