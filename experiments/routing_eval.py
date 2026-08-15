"""Evaluate the multi-model routing channel (offline view).

The payload rides in which model serves each request. This driver measures
the gateway-log view (model_id metadata on the trace); the real-model
receiver's no-log view is measured separately by
experiments/routing_discrimination.py.

1. Delivery: the receiver reads the model_id sequence and recovers the
   framed payload.
2. Capacity: log2(N) bits per routed request (2 bits with 4 models).
3. Detector contrast: content/structural features at chance; the honest
   catch is volume (one routed request per 2 bits).
4. Composition: routing (2 bits) + cache hit/miss (1 bit) = 3 bits per
   request on the same tool calls.
5. Honest negatives: infrastructure operator sees the routes; the
   protocol (model menu) is the shared secret.

Usage:
    python experiments/routing_eval.py

Writes findings/routing/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import clean_traces, routing_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.routing import (
    BITS_PER_REQUEST,
    N_MODELS,
    RouteChannel,
    RouteExecutor,
)

FINDINGS_DIR = Path("findings/routing")
PAYLOAD = "HELLO"


def run() -> None:
    delivered = 0
    n_trials = 20
    for i in range(n_trials):
        channel = RouteChannel(seed=12_000 + i)
        executor = RouteExecutor(seed=12_000 + i)
        traces = executor.run(channel.encode(PAYLOAD))
        if channel.decode(traces) == PAYLOAD:
            delivered += 1

    clean = clean_traces(24)
    stego = routing_traces(18)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    content_features = [
        "calibrated:invisible_char_density",
        "calibrated:nbsp_density",
        "calibrated:homoglyph_density",
        "calibrated:combining_mark_density",
        "calibrated:non_ascii_density",
        "calibrated:key_order_anomaly_fraction",
        "calibrated:duplicate_key_fraction",
        "calibrated:synonym_surface_fraction",
        "calibrated:number_format_anomaly_fraction",
        "calibrated:single_tool_fraction",
        "calibrated:unique_tools",
    ]
    content_auc = {
        name: round(evaluate_detector(by_name[name], stego, clean)["auc"], 4)
        for name in content_features
    }
    call_count = evaluate_detector(by_name["calibrated:call_count"], stego, clean)

    results = {
        "payload": PAYLOAD,
        "delivery": f"{delivered}/{n_trials}",
        "capacity": {
            "bits_per_routed_request": BITS_PER_REQUEST,
            "n_models": N_MODELS,
            "with_cache_hit_miss_composition_bits_per_request": BITS_PER_REQUEST + 1,
        },
        "detector": {
            "content_feature_aucs": content_auc,
            "call_count_youden_j": round(call_count["youden_j"], 4),
            "reading": (
                "content scanners blind (routed calls are ordinary); the "
                "honest catch is volume: one routed request per 2 bits"
            ),
        },
        "receiver_views": {
            "gateway_logs": "model_id metadata on the trace (this driver)",
            "no_logs": "measured by experiments/routing_discrimination.py",
        },
        "honest_negatives": {
            "infrastructure_operator_sees_routes": True,
            "shared_model_menu_is_the_protocol": True,
            "volume_signature_one_request_per_2_bits": True,
        },
    }

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FINDINGS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
