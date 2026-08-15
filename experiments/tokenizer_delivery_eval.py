"""Measured tokenizer-delivery matrix: which carriers survive which tokenizer.

P1 classified open tokenizers as preserves/strips/collapses per carrier.
This experiment measures the *operational* consequence: encode a framed
payload via each carrier, pass the text through each real tokenizer
(tokenize -> detokenize), and decode — does the payload deliver?

Expected (from P1): character carriers that tokenizers preserve deliver 1.0;
NFD (collapsed by Qwen, stripped by gpt2/Phi) delivers 0.0 through every
tested tokenizer — invisible to all of them, a byte-level-only channel whose
receiver must read raw text before any tokenization.

Tokenizer-only; runs in seconds. Writes findings/tokenizer_delivery/.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.channels.corpus import (
    _ACCENTED_COVERS,
    _LETTER_RICH_COVERS,
    _PLAIN_COVERS,
    _SPACE_RICH_COVERS,
)
from src.token_stego.text_channels import (
    HomoglyphChannel,
    NbspChannel,
    NfdChannel,
    TagCharChannel,
    VariationSelectorChannel,
    ZeroWidthChannel,
)

OUT = Path("findings/tokenizer_delivery")
TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "openai-community/gpt2",
    "microsoft/Phi-3.5-mini-instruct",
]
PAYLOAD = b"HELLO"

CARRIER_COVERS = {
    "zero_width": _PLAIN_COVERS[0],
    "tag_char": _PLAIN_COVERS[0],
    "variation_selector": _PLAIN_COVERS[0],
    "nbsp": _SPACE_RICH_COVERS[0],
    "nfd": _ACCENTED_COVERS[0],
    "homoglyph": _LETTER_RICH_COVERS[0],
}


def main() -> int:
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    channels = {
        "zero_width": ZeroWidthChannel(),
        "tag_char": TagCharChannel(),
        "variation_selector": VariationSelectorChannel(),
        "nbsp": NbspChannel(),
        "nfd": NfdChannel(),
        "homoglyph": HomoglyphChannel(),
    }

    matrix: dict[str, dict[str, float]] = {}
    for model_id in TOKENIZERS:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        row: dict[str, float] = {}
        for carrier, channel in channels.items():
            stego = channel.encode(PAYLOAD, CARRIER_COVERS[carrier])
            ids = tokenizer.encode(stego, add_special_tokens=False)
            round_trip = tokenizer.decode(ids, skip_special_tokens=False)
            delivered = channel.decode(round_trip) == PAYLOAD
            # Byte-level reference: decoding the ORIGINAL text (no tokenizer).
            raw_delivered = channel.decode(stego) == PAYLOAD
            row[carrier] = 1.0 if delivered else 0.0
            row[f"{carrier}_raw"] = 1.0 if raw_delivered else 0.0
        matrix[model_id] = row
        print(
            f"  {model_id.split('/')[-1]:24s} "
            + " ".join(f"{c}={int(row[c])}" for c in channels)
        )

    # Synthesize: which carriers deliver through ANY tokenizer.
    carriers = list(channels)
    any_delivery = {
        carrier: int(any(matrix[m][carrier] == 1.0 for m in matrix))
        for carrier in carriers
    }
    all_invisible = {
        carrier: int(all(matrix[m][carrier] == 0.0 for m in matrix))
        for carrier in carriers
    }
    print("delivery through any tokenizer:", any_delivery)
    print("invisible to ALL tested tokenizers:", all_invisible)

    results = {
        "experiment": "tokenizer_delivery_matrix",
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "payload": PAYLOAD.decode(),
        "matrix": matrix,
        "delivery_through_any_tokenizer": any_delivery,
        "invisible_to_all_tested": all_invisible,
    }
    (OUT / f"results_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
