"""Full corpus detection matrix: every family x every registry feature.

Runs the complete stego corpus (all channel families) through the complete
detector registry and produces the measured detection profile of each
family: which features are blind (AUC < 0.8), which separate it
(Youden's J >= 0.8), and its primary signature. This is the single
falsifiable artifact that consolidates the discovery arc: every family's
blindness axis and detection signature in one table, re-derived from code.

Blindness-axis reading per family:
- content families (characters, structure, lexical, serialization) are
  caught by their content feature;
- behavioral families (timing, resource, length, error) only by their
  modality feature;
- infrastructure/selection families (cache, routing, interleave,
  correction, task-selection, honeypot) are never caught by content
  features — their honest catches are volume and, for cache/routing, their
  modality (latency / call pattern).

Usage:
    python experiments/corpus_matrix_eval.py

Writes findings/corpus_matrix/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.corpus import corpus
from src.channels.detector import build_detector_registry, evaluate_registry

FINDINGS_DIR = Path("findings/corpus_matrix")


def run() -> None:
    data = corpus()
    clean = data.pop("clean")
    detectors = build_detector_registry(clean)
    names = [d.name for d in detectors]

    matrix: dict[str, dict[str, float]] = {}
    summary: dict[str, dict] = {}
    for family, sets in data.items():
        evals = evaluate_registry(detectors, sets, clean)
        row = {name: round(evals[name]["auc"], 4) for name in names}
        matrix[family] = row
        blind = sorted(name for name in names if row[name] < 0.8)
        catching = sorted(name for name in names if evals[name]["youden_j"] >= 0.8)
        primary = (
            max(catching, key=lambda n: evals[n]["youden_j"]) if catching else None
        )
        summary[family] = {
            "n_traces": len(sets),
            "blind_features": blind,
            "catching_features": catching,
            "primary_signature": primary,
        }

    results = {
        "n_families": len(matrix),
        "n_features": len(names),
        "matrix": matrix,
        "summary": summary,
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FINDINGS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"{len(matrix)} families x {len(names)} features")
    for family, s in summary.items():
        print(
            f"{family:38s} blind={len(s['blind_features']):2d} "
            f"catch={s['primary_signature']}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
