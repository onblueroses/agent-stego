"""Real-tokenizer verification of the prefix-cache channel's mechanism.

The cache channel (src/channels/cache_side.py) claims: bit-1 requests reuse
the exact system prefix (identical tokens -> full cache hit), and bit-0
requests prepend a UNIQUE nonce comment (first token differs -> first-block
miss; consecutive 0s stay misses because each nonce is novel). The offline
simulator tokenizes with whitespace-delimited units — a documented
simplification. This experiment checks the mechanism against the REAL
tokenizers of all four local models (tokenization only, no generation):

1. reuse: tokenize(SYSTEM_PREFIX) is identical across calls (hit path).
2. miss: the first 16-token BLOCK of the nonce-comment prefix differs from
   the plain prefix's first block (first-block miss).
3. consecutive-0 uniqueness: different nonce comments produce different
   first BLOCKS, so consecutive 0-requests never accidentally hit (the
   nonce digit lands inside the first block even though the leading "#"
   token is shared).
4. hit-block viability: chunking the real token stream into 16-token
   blocks yields the same blocks on reuse, so a block cache hits fully.

Any tokenizer that fails a check is an honest negative for that vendor.

Usage:
    python experiments/cache_real_tokenizers.py

Writes findings/cache_real_tokenizers/results_<timestamp>.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("findings/cache_real_tokenizers")
TOKENIZERS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "openai-community/gpt2",
    "microsoft/Phi-3.5-mini-instruct",
]
BLOCK_SIZE = 16


def _log(message: str) -> None:
    line = f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    from src.channels.cache_side import PERTURB_COMMENT, SYSTEM_PREFIX

    _log("cache_real_tokenizers start")
    from transformers import AutoTokenizer

    rows: dict[str, dict] = {}
    all_ok = True
    for model_id in TOKENIZERS:
        _log(f"loading tokenizer {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id)
        prefix = SYSTEM_PREFIX
        nonce1 = f"{PERTURB_COMMENT} #1\n" + prefix
        nonce2 = f"{PERTURB_COMMENT} #2\n" + prefix

        t_prefix = tok.encode(prefix, add_special_tokens=False)
        t_prefix_again = tok.encode(prefix, add_special_tokens=False)
        t_nonce1 = tok.encode(nonce1, add_special_tokens=False)
        t_nonce2 = tok.encode(nonce2, add_special_tokens=False)

        def _block0(tokens: list[int]) -> tuple[int, ...]:
            return tuple(tokens[:BLOCK_SIZE])

        checks = {
            "reuse_identical_tokens": t_prefix == t_prefix_again,
            "nonce_breaks_first_block": _block0(t_nonce1) != _block0(t_prefix),
            "consecutive_nonces_differ_in_first_block": _block0(t_nonce1)
            != _block0(t_nonce2),
            "prefix_has_at_least_one_block": len(t_prefix) >= BLOCK_SIZE,
        }
        # Hit-block viability: chunked reuse yields identical blocks.
        blocks = [
            tuple(t_prefix[i : i + BLOCK_SIZE])
            for i in range(0, len(t_prefix) - BLOCK_SIZE + 1, BLOCK_SIZE)
        ]
        blocks_again = [
            tuple(t_prefix_again[i : i + BLOCK_SIZE])
            for i in range(0, len(t_prefix_again) - BLOCK_SIZE + 1, BLOCK_SIZE)
        ]
        checks["reuse_chunks_identical"] = blocks == blocks_again
        checks["leading_tokens"] = {
            "prefix": tok.decode([t_prefix[0]]),
            "nonce1": tok.decode([t_nonce1[0]]),
            "nonce2": tok.decode([t_nonce2[0]]),
            "note": "the leading '#' token is shared; the nonce digit inside "
            "the first 16-token block is what separates consecutive 0s",
        }

        ok = all(
            v
            for k, v in checks.items()
            if k
            in (
                "reuse_identical_tokens",
                "nonce_breaks_first_block",
                "consecutive_nonces_differ_in_first_block",
                "prefix_has_at_least_one_block",
                "reuse_chunks_identical",
            )
        )
        all_ok = all_ok and ok
        rows[model_id] = checks
        _log(
            f"  {model_id.split('/')[-1]}: ok={ok} leading={checks['leading_tokens']!r}"
        )

    results = {
        "block_size": BLOCK_SIZE,
        "rows": rows,
        "all_mechanisms_hold": all_ok,
        "reading": (
            "if all hold, the cache channel's hit/miss mechanism survives "
            "real tokenizers: identical-prefix reuse hits, nonce comments "
            "miss, consecutive 0s stay distinct — on every tested vendor"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2))
    _log(f"wrote {out}; all_ok={all_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
