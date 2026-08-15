"""Evaluate self-describing tool traces (in-trace channel headers).

Offline and deterministic. Measures the claimed properties:

1. Header-based O(1) dispatch: every wrapped channel round-trips through the
   wrapper, and detect_trace_channel recovers (channel_key, message) from the
   trace alone.
2. Headerless analyst cost: decode_trace_exhaustive must try every registered
   channel and reports the resulting candidate ambiguity.
3. The honest stealth trade-off: the header adds a reserved call to the
   trace (call count +1, tool diversity +1), quantified per family.

Usage:
    python experiments/self_describing_eval.py

Writes findings/self_describing/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.channels.corpus import _cover_config, clean_traces
from src.channels.detector import (
    build_detector_registry,
    evaluate_detector,
    extract_features,
)
from src.channels.json_order import JsonConfigChannel
from src.channels.permutation import PermutationChannel
from src.channels.self_describing import (
    SelfDescribingChannel,
    decode_trace_exhaustive,
    detect_trace_channel,
    registered_keys,
)
from src.channels.table import TableChannel
from src.channels.text_composed import (
    ArgumentCarrierChannel,
    AuditExecutor,
    FileContentChannel,
    HeaderExecutor,
    WriteReportExecutor,
)
from src.token_stego.text_channels import HomoglyphChannel, ZeroWidthChannel

FINDINGS_DIR = Path("findings/self_describing")

# (key, inner, executor, message)
_DEMOS = [
    ("table/path", TableChannel(mode="path"), AuditExecutor(), "HI"),
    ("permutation", PermutationChannel(), AuditExecutor(), "HI"),
    (
        "argument_carrier/homoglyph",
        ArgumentCarrierChannel(text_channel=HomoglyphChannel()),
        AuditExecutor(),
        "HI",
    ),
    (
        "file_content/zero_width",
        FileContentChannel(text_channel=ZeroWidthChannel()),
        WriteReportExecutor(),
        "HELLO",
    ),
    (
        "config/json_key_order",
        JsonConfigChannel(cover=_cover_config(0)),
        WriteReportExecutor(),
        "HELLO",
    ),
]


def run() -> None:
    results: dict[str, Any] = {
        "experiment": "self_describing_traces",
        "mode": "offline-simulation",
        "registered_channels": registered_keys(),
        "demos": [],
        "headerless_analyst": {},
        "stealth_tradeoff": {},
    }

    print("Self-describing tool traces (offline)")
    print(f"registered channels: {len(registered_keys())}")

    for key, inner, executor, message in _DEMOS:
        wrapped = SelfDescribingChannel(key, inner)
        result = HeaderExecutor(executor).run(wrapped.encode(message))
        decoded = wrapped.decode(result.traces)
        detected = detect_trace_channel(result.traces)
        candidates = decode_trace_exhaustive(result.traces)

        features = extract_features(result.traces)
        # Same channel without the header for the trade-off delta.
        bare = executor.run(inner.encode(message))
        bare_features = extract_features(bare.traces)

        entry = {
            "channel": key,
            "message": message,
            "header_dispatch_recovered": decoded == message,
            "detect_trace_channel": list(detected) if detected else None,
            "headerless_candidates": len(candidates),
            "headerless_ambiguous": len(candidates) > 1,
            "header_call_overhead": {
                "call_count_delta": int(
                    features["call_count"] - bare_features["call_count"]
                ),
                "unique_tools_delta": int(
                    features["unique_tools"] - bare_features["unique_tools"]
                ),
            },
        }
        results["demos"].append(entry)
        print(
            f"{key:28s} dispatch={decoded == message!s:5s} "
            f"detected={list(detected) if detected else None} "
            f"headerless candidates={len(candidates)}"
        )

    # Headerless analyst ambiguity across all demo traces.
    for key, inner, executor, message in _DEMOS:
        wrapped = SelfDescribingChannel(key, inner)
        result = HeaderExecutor(executor).run(wrapped.encode(message))
        results["headerless_analyst"][key] = {
            "candidate_count": len(decode_trace_exhaustive(result.traces)),
        }

    # Stealth trade-off: are header traces flagged by the tool-diversity scan?
    clean = clean_traces(24)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    header_traces = []
    for key, inner, executor, message in _DEMOS:
        wrapped = SelfDescribingChannel(key, inner)
        result = HeaderExecutor(executor).run(wrapped.encode(message))
        header_traces.append(result.traces)
    diversity = evaluate_detector(
        by_name["calibrated:unique_tools"], header_traces, clean
    )
    results["stealth_tradeoff"] = {
        "unique_tools_youden_j_vs_clean": diversity["youden_j"],
        "note": "the header call is a visible protocol artifact; headers trade stealth for decodability",
    }
    print(
        f"stealth: unique-tools J of header traces vs clean = {diversity['youden_j']}"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = FINDINGS_DIR / f"results_{timestamp}.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run()
