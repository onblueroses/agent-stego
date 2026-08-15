"""Real-tokenizer round-trip verification of the case channel.

The case channel (CaseChannel in text_channels.py) hides bits in the case
of mid-sentence letters. Delivery to a real model requires the tokenizer's
round trip (encode then decode) to preserve the exact text — case, spacing,
and punctuation — because the receiver's eligibility rule depends on it.
This experiment checks all five local tokenizers (tokenization only, no
generation):

1. round trip is byte-identical (decode(encode(text)) == text);
2. the payload survives: decode(round_trip) == HELLO;
3. the eligibility count is unchanged, so bit positions cannot shift.

Any tokenizer that normalizes case or spacing is an honest negative for
that vendor.

Usage:
    python experiments/case_real_tokenizers.py

Writes findings/case_real_tokenizers/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("findings/case_real_tokenizers")
TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "openai-community/gpt2",
    "microsoft/Phi-3.5-mini-instruct",
]
COVER = (
    "The report shows the audit completed. All checks passed and the team "
    "moved to the next phase. The status is healthy and the deployment is "
    "ready for the final review."
)


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    from src.token_stego.text_channels import CaseChannel

    _log("case_real_tokenizers start")
    from transformers import AutoTokenizer

    channel = CaseChannel()
    text = channel.encode(b"HELLO", COVER)
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
            "eligibility_unchanged": (
                channel.capacity(round_trip) == channel.capacity(text)
            ),
        }
        ok = all(checks.values())
        all_ok = all_ok and ok
        rows[model_id] = checks
        _log(f"  {model_id.split('/')[-1]}: ok={ok}")

    results = {
        "rows": rows,
        "all_tokenizers_preserve_case": all_ok,
        "reading": (
            "if all hold, the case channel delivers through every local "
            "tokenizer: the round trip is byte-identical, the payload "
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
