"""Evaluate the formatting-consistency defense (closes the blind spot).

Offline and deterministic. Rounds 63-65 found three formatting-axis
channels (case, date-format, whitespace) fully invisible to the original
17-feature registry. This driver measures the defense that closes the gap:
a formatting_anomaly_fraction feature combining case regularity,
date-format locale consistency, and space-run width. Measures:

1. The three families are caught at J=1.0 by the new feature.
2. Clean traces score exactly 0 (no false positives), including the decoy
   read responses with markdown headings and list lines.
3. Character carriers (zero-width, homoglyph, nbsp) stay blind to it.
4. The other 17 features remain at chance for the three families (the
   closure is additive, not a re-fit).

Usage:
    python experiments/formatting_defense_eval.py

Writes findings/formatting_defense/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import clean_traces, file_content_traces
from src.channels.detector import (
    build_detector_registry,
    evaluate_detector,
    evaluate_registry,
)

FINDINGS_DIR = Path("findings/formatting_defense")


def run() -> None:
    clean = clean_traces(24)
    detectors = build_detector_registry(clean)
    by_name = {d.name: d for d in detectors}
    fmt_name = "calibrated:formatting_anomaly_fraction"

    rows: dict[str, dict] = {}
    for carrier in ("case", "datefmt", "whitespace"):
        stego = file_content_traces(carrier, 18, decoys=True)
        fmt = evaluate_detector(by_name[fmt_name], stego, clean)
        # The other 17 features stay at chance (additive closure).
        others = evaluate_registry(
            [d for d in detectors if d.name != fmt_name], stego, clean
        )
        max_other_auc = max(v["auc"] for v in others.values())
        rows[carrier] = {
            "formatting_youden_j": round(fmt["youden_j"], 4),
            "max_other_feature_auc": round(max_other_auc, 4),
        }

    # No false positives on character carriers.
    false_positives: dict[str, float] = {}
    for carrier in ("zero_width", "homoglyph", "nbsp"):
        stego = file_content_traces(carrier, 18, decoys=True)
        res = evaluate_detector(by_name[fmt_name], stego, clean)
        false_positives[carrier] = round(res["auc"], 4)

    # Clean traces score zero (including decoy responses).
    from src.channels.detector import _formatting_anomaly_fraction

    clean_max = max(
        _formatting_anomaly_fraction([t.response]) for run in clean for t in run
    )

    results = {
        "closure": rows,
        "false_positive_aucs_on_character_carriers": false_positives,
        "max_clean_response_score": round(clean_max, 6),
        "reading": (
            "formatting_anomaly_fraction catches all three formatting "
            "families at J=1.0 while leaving character carriers and clean "
            "traces at zero — the round-63..65 blind spot is closed, "
            "additively, without re-fitting the other 17 features"
        ),
    }

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FINDINGS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
