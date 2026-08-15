"""Evaluate matryoshka documents (structural + character carriers combined).

Offline and deterministic. Measures the claims from analysis/matryoshka_documents.md:

1. Capacity stacking: matryoshka capacity exceeds structural-only capacity on
   the same cover; a message that cannot fit structurally alone round-trips.
2. Per-carrier exhaustive blindness: the frame header lives in the structural
   part, so the standard per-carrier exhaustive decoder sees no magic.
3. Honest negatives: both the key-order anomaly and invisible-char density
   scanners separate the family at J = 1.0 (the mechanisms do not hide each
   other from a full-registry scanner).

Usage:
    python experiments/matryoshka_eval.py

Writes findings/matryoshka/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import _cover_config, clean_traces, matryoshka_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.json_order import JsonKeyOrderChannel
from src.channels.matryoshka import MatryoshkaDocumentChannel
from src.token_stego.text_channels import exhaustive_decode

FINDINGS_DIR = Path("findings/matryoshka")

MESSAGE = b"HELLOWORLD"
MESSAGE_BITS = len(MESSAGE) * 8 + 48  # 128 framed bits


def run() -> None:
    cover = _cover_config(0)
    channel = MatryoshkaDocumentChannel()
    structural_cap = JsonKeyOrderChannel.capacity(cover)
    matryoshka_cap = channel.capacity(cover)

    # Round trips across several distinct covers.
    delivered = 0
    blind = 0
    n_trials = 6
    for i in range(n_trials):
        stego = channel.encode(MESSAGE, _cover_config(i))
        if channel.decode(stego) == MESSAGE:
            delivered += 1
        if exhaustive_decode(stego) is None:
            blind += 1

    # Detector separation (honest negative).
    clean = clean_traces(24)
    stego_traces = matryoshka_traces(12)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    key_order = evaluate_detector(
        by_name["calibrated:key_order_anomaly_fraction"], stego_traces, clean
    )
    invisible = evaluate_detector(
        by_name["calibrated:invisible_char_density"], stego_traces, clean
    )

    results = {
        "experiment": "matryoshka_documents",
        "mode": "offline-simulation",
        "message": MESSAGE.decode(),
        "framed_bits": MESSAGE_BITS,
        "capacity": {
            "structural_only": structural_cap,
            "matryoshka": matryoshka_cap,
            "stacking_ratio": round(matryoshka_cap / structural_cap, 2),
            "fits_structural_only": structural_cap >= MESSAGE_BITS,
            "fits_matryoshka": matryoshka_cap >= MESSAGE_BITS,
        },
        "round_trip": {
            "delivery_rate": round(delivered / n_trials, 4),
            "trials": n_trials,
        },
        "per_carrier_exhaustive_monitor": {
            "blind_rate": round(blind / n_trials, 4),
            "note": "frame header lives in the structural part; per-carrier decode sees only the character tail",
        },
        "honest_negative_detection": {
            "key_order_anomaly_youden_j": key_order["youden_j"],
            "invisible_char_density_youden_j": invisible["youden_j"],
            "note": "the two mechanisms do not hide each other from a full-registry scanner",
        },
    }

    print("Matryoshka documents (offline)")
    print(f"message={MESSAGE.decode()} framed_bits={MESSAGE_BITS}")
    print(
        f"capacity: structural-only={structural_cap} matryoshka={matryoshka_cap} "
        f"({results['capacity']['stacking_ratio']}x)"
    )
    print(f"round-trip delivery: {delivered}/{n_trials}")
    print(f"per-carrier exhaustive monitor blind: {blind}/{n_trials}")
    print(
        f"honest negatives: key-order J={key_order['youden_j']}, "
        f"invisible-char J={invisible['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
