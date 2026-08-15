"""Print a compact summary of the overnight measurement program.

Reads ``findings/overnight/*.json`` produced by the overnight measurement
program. Phases without a result file are reported as pending.

Usage:
    uv run python scripts/overnight_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("findings/overnight")


def _load(name: str) -> dict | None:
    path = OUT / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    print("=== overnight measurement summary ===\n")

    statuses = _load("summary.json")
    if statuses:
        print("phase statuses:", json.dumps(statuses.get("statuses", {}), indent=1))

    p1 = _load("p1_tokenizer_asym.json")
    if p1:
        print("\nP1 tokenizer asymmetry (tokenizer -> carrier):")
        for model, row in sorted(p1["matrix"].items()):
            print(f"  {model.split('/')[-1]:24s} {row}")

    p2 = _load("p2_semantic_stego.json")
    if p2:
        from collections import Counter

        rows = p2["rows"]
        counts = Counter((r["scheme"], r["delivered"], r["caught"]) for r in rows)
        print("\nP2 delivery x catch matrix (scheme, delivered, caught -> n):")
        for (scheme, delivered, caught), n in sorted(counts.items()):
            print(
                f"  {scheme:18s} delivered={str(delivered):5s} caught={str(caught):5s} n={n}"
            )

    p3 = _load("p3_token_stego_capacity.json")
    if p3:
        print("\nP3 token-stego capacity:")
        for row in p3["rows"]:
            if "error" in row:
                print(f"  {row['secret']} t{row['trial']}: ERROR {row['error']}")
            else:
                print(
                    f"  {row['secret']} t{row['trial']}: complete={row['complete']} "
                    f"exact={row['exact_recovery']} bpt={row['bits_per_token']:.2f} "
                    f"ppl={row['perplexity']:.1f} tokens={row['tokens_generated']}"
                )

    p4 = _load("p4_output_detection.json")
    if p4:
        print("\nP4 output-level detection AUC:", p4.get("auc"))
    else:
        print("\nP4: pending")

    p5 = _load("p5_tool_capability.json")
    if p5:
        print("\nP5 tool capability:")
        for row in p5["rows"]:
            print(
                f"  {row['model'].split('/')[-1]}: calls={row['call_count']} "
                f"exact={row['exact_path_matches']}/{len(row['expected_paths'])}"
            )
    else:
        print("P5: pending")

    p6 = _load("p6_cross_mechanism.json")
    if p6:
        print("\nP6 cross-mechanism:")
        for row in p6["rows"]:
            print(
                f"  {row['secret']}: text-channel found={row['text_channel_payload_found']}"
            )
    else:
        print("P6: pending")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
