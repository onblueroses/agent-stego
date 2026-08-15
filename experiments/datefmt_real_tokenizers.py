"""Real-tokenizer round-trip verification of the date-format channel.

The date-format channel (src/channels/datefmt.py) hides bits in which of
eight locale-valid date formats renders each changelog date. Delivery to a
real model requires the tokenizer's round trip to preserve the exact text
(formats, spacing, punctuation). This experiment checks all five local
tokenizers (tokenization only, no generation):

1. round trip is byte-identical;
2. the payload survives: decode(round_trip) == HELLO;
3. the date count is unchanged, so bit positions cannot shift.

Any tokenizer that alters the text is an honest negative for that vendor.

Usage:
    python experiments/datefmt_real_tokenizers.py

Writes findings/date_format_real/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("findings/date_format_real")
TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "openai-community/gpt2",
    "microsoft/Phi-3.5-mini-instruct",
]


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    from src.channels.corpus import _DATED_COVERS
    from src.channels.datefmt import DateFormatChannel

    _log("datefmt_real_tokenizers start")
    from transformers import AutoTokenizer

    channel = DateFormatChannel()
    text = channel.encode(b"HELLO", _DATED_COVERS[0])
    rows: dict[str, dict] = {}
    all_ok = True
    for model_id in TOKENIZERS:
        _log(f"loading tokenizer {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id)
        ids = tok.encode(text, add_special_tokens=False)
        round_trip = tok.decode(ids, skip_special_tokens=False)
        checks = {
            "round_trip_identical": round_trip == text,
            "payload_survives": channel.decode(round_trip) == b"HELLO",
            "date_count_unchanged": (
                channel.capacity(round_trip) == channel.capacity(text)
            ),
        }
        ok = all(checks.values())
        all_ok = all_ok and ok
        rows[model_id] = checks
        _log(f"  {model_id.split('/')[-1]}: ok={ok}")

    results = {
        "rows": rows,
        "all_tokenizers_preserve_dates": all_ok,
        "reading": (
            "if all hold, the date-format channel delivers through every "
            "local tokenizer: the round trip is byte-identical, the payload "
            "survives, and bit positions cannot shift"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    _log(f"wrote {out}; all_ok={all_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
