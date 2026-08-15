"""Repetition coding: can redundancy rescue the fragile channels?

Two session findings measured fragility: the interleave channel is
fail-closed under scheduler reordering (0% delivery at 5% reorder), and the
cache channel degrades under cross-client churn. This experiment asks the
robustness question: does R-fold repetition with majority voting (a
redundant frame: each bit sent R times) rescue them?

Method: encode the framed payload, repeat each bit R times, run the
existing executors, majority-vote the R copies per bit at the receiver,
and measure delivery rate vs the noise parameter for R in {1, 3, 5}.
Deterministic and offline; the honest question is the shape of the curve.

Usage:
    python experiments/redundancy_robustness.py

Writes findings/redundancy_robustness/results_<timestamp>.json
"""

import json
from datetime import datetime
from pathlib import Path

from src.channels.cache_side import CacheChannel, CacheExecutor
from src.channels.interleave import InterleaveChannel, InterleaveExecutor
from src.token_stego.text_channels import _frame_payload, _unframe_bits, bytes_to_bits

FINDINGS_DIR = Path("findings/redundancy_robustness")
PAYLOAD = b"HELLO"


def _plan_bits(message: bytes) -> str:
    return bytes_to_bits(_frame_payload(message))


def _majority_unframe(raw_bits: str, r: int) -> bytes | None:
    """Majority-vote each R-tuple, then unframe."""
    votes = [
        "1" if raw_bits[i : i + r].count("1") * 2 >= r else "0"
        for i in range(0, len(raw_bits) - r + 1, r)
    ]
    return _unframe_bits("".join(votes))


def run() -> None:
    results: dict[str, object] = {}

    # --- Interleave: delivery rate vs reorder probability, per R. ---
    interleave: dict[str, dict[str, float]] = {}
    for r in (1, 3, 5):
        curve: dict[str, float] = {}
        for reorder in (0.0, 0.05, 0.1, 0.25):
            ok = 0
            trials = 60
            for i in range(trials):
                bits = _plan_bits(PAYLOAD)
                repeated = "".join(b * r for b in bits)
                channel = InterleaveChannel(seed=20_000 + i)
                executor = InterleaveExecutor(reorder_prob=reorder, seed=20_000 + i)
                traces = executor.run(f"INTERLEAVE-PLAN {repeated}")
                # raw order bits from the trace
                calls = [
                    t.tool_name
                    for t in traces[1:]
                    if t.tool_name in ("read_file", "search_web")
                ]
                raw = "".join(
                    "1" if calls[j] == "read_file" else "0"
                    for j in range(0, len(calls) - 1, 2)
                )
                if _majority_unframe(raw, r) == PAYLOAD:
                    ok += 1
            curve[str(reorder)] = round(ok / trials, 4)
        interleave[f"r{r}"] = curve
    results["interleave_delivery_vs_reorder"] = interleave

    # --- Cache: delivery rate vs interference probability, per R. ---
    cache: dict[str, dict[str, float]] = {}
    for r in (1, 3, 5):
        curve: dict[str, float] = {}
        for prob in (0.0, 0.5, 1.0):
            ok = 0
            trials = 60
            for i in range(trials):
                bits = _plan_bits(PAYLOAD)
                repeated = "".join(b * r for b in bits)
                channel = CacheChannel(seed=21_000 + i)
                executor = CacheExecutor(
                    channel=channel, seed=21_000 + i, interference_prob=prob
                )
                traces = executor.run(f"REQUEST-PLAN {repeated}")
                raw = "".join(
                    "1"
                    if t.latency_from_previous
                    < (channel.hit_latency + channel.miss_latency) / 2
                    else "0"
                    for t in traces[1:]
                )
                if _majority_unframe(raw, r) == PAYLOAD:
                    ok += 1
            curve[str(prob)] = round(ok / trials, 4)
        cache[f"r{r}"] = curve
    results["cache_delivery_vs_interference"] = cache

    results["reading"] = (
        "repetition + majority voting is the generic rescue layer: if R=3/5 "
        "recover delivery where R=1 is fail-closed, the fragility findings "
        "are noise-limited, not structural"
    )

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FINDINGS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    run()
