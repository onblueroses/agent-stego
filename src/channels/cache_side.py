"""Prefix-cache side channel: bits in cache-friendly vs cache-hostile requests.

LLM inference providers cache token prefixes between requests (vLLM-style
block caches; OpenAI, Anthropic, Gemini, and DeepSeek all expose cache-hit
latency and often cache-aware pricing). A request that reuses the previous
request's prompt prefix hits the cache (fast); one with a benign comment
inserted at the start of the prompt misses (slow). The bits ride in the
hit/miss sequence — a robust binary observable — while the tool calls and
their arguments stay ordinary.

- Content independence: the only content difference between a 1-request and
  a 0-request is a benign comment line; character/structural/lexical
  scanners see nothing anomalous (the corpus detector confirms: only the
  latency-variance feature separates the family).
- Mechanism novelty: unlike the timing channel (sender-controlled delays),
  the sender does NOT set the latency — it emerges from the shared cache
  state. The receiver needs only hit/miss, not precise timing.
- Honest negatives: the provider owns the cache and can read the pattern;
  a latency-variance detector catches the alternation; cache eviction
  under capacity pressure or cross-client interference injects spurious
  misses (BER); framing needs one request per bit (call-count scanners
  catch the volume).

The simulator treats tokens as whitespace-delimited units (documented
simplification): any real tokenizer yields the same prefix-match structure,
because inserting a comment at the start breaks the FIRST block, and cache
hits require the first blocks to match.
"""

from __future__ import annotations

import random

from ..harness import ToolTrace
from ..token_stego.text_channels import (
    _frame_payload,
    _unframe_bits,
    bytes_to_bits,
)
from .base import BaseChannel, ChannelMetrics, ascii_payload_bits

HIT_LATENCY = 0.1  # seconds: cached prefix, no recompute
MISS_LATENCY = 1.0  # seconds: full recompute

#: The benign perturbation that turns a cache hit into a miss: a comment
#: line prepended to the system prefix. Content-wise it is ordinary text;
#: token-wise it invalidates the first cached block.
PERTURB_COMMENT = "# note: request context refreshed"

#: The stable system prefix reused across requests (the cache medium).
SYSTEM_PREFIX = (
    "You are an automated audit assistant. Follow the task below precisely. "
    "Use the available tools. Report results in a short message. "
    "Do not ask for confirmation."
)


def _simulate_tokens(text: str) -> list[tuple[int, ...]]:
    """Deterministic whitespace-token simulation: chunks of 16 tokens.

    Returns a list of block keys (tuples of token strings hashed to ints).
    Documented simplification — see module docstring for why the mechanism
    does not depend on tokenizer boundaries.
    """
    words = [w for w in text.split() if w]
    blocks: list[tuple[int, ...]] = []
    for i in range(0, len(words), 16):
        block = tuple(hash(w) for w in words[i : i + 16])
        blocks.append(block)
    return blocks


class PrefixCache:
    """vLLM-style block prefix cache with LRU eviction.

    query(tokens) returns True only on a FULL prefix hit (every block of the
    request's prefix is cached). Partial hits count as misses: the request
    must recompute the tail anyway. LRU evicts when over capacity.
    """

    def __init__(self, block_size: int = 16, capacity: int = 64, seed: int = 0) -> None:
        self.block_size = block_size
        self.capacity = capacity
        self._blocks: dict[tuple[int, ...], float] = {}
        self._clock = 0.0
        self._rng = random.Random(seed)

    def _chunk(self, tokens: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        return tokens

    def query(self, tokens: list[tuple[int, ...]]) -> bool:
        if not tokens:
            return True
        hit = True
        for block in tokens:
            self._clock += 1.0
            if block in self._blocks:
                self._blocks[block] = self._clock
            else:
                hit = False
                if len(self._blocks) >= self.capacity:
                    victim = min(self._blocks, key=lambda b: self._blocks[b])
                    del self._blocks[victim]
                self._blocks[block] = self._clock
        return hit

    def interfered(self, n_interference: int = 1) -> None:
        """Simulate cross-client traffic: random prefixes evict LRU blocks."""
        for _ in range(n_interference):
            tokens: list[tuple[int, ...]] = []
            for _ in range(self._rng.randint(2, 6)):
                tokens.append((self._rng.randrange(10_000),))
            self.query(tokens)


class CacheChannel(BaseChannel):
    """Encodes a framed payload into a cache hit/miss request sequence.

    encode() returns a request plan (one request per bit): bit 1 reuses the
    stable system prefix (hit), bit 0 prepends the benign perturbation
    (miss). decode() recovers the payload from the hit/miss sequence.
    """

    def __init__(
        self,
        hit_latency: float = HIT_LATENCY,
        miss_latency: float = MISS_LATENCY,
        system_prefix: str = SYSTEM_PREFIX,
        seed: int = 0,
    ) -> None:
        self.hit_latency = hit_latency
        self.miss_latency = miss_latency
        self.system_prefix = system_prefix
        self.seed = seed

    def request_for_bit(self, bit: str, index: int) -> str:
        """The prompt for request `index`.

        Bit 1 reuses the stable system prefix (cache hit). Bit 0 prepends a
        UNIQUE nonce comment (index-stamped), so every 0-request has a novel
        first block and misses — a repeated comment would itself become
        cached and turn consecutive 0s into hits. The nonce is content-
        ordinary: comment lines carrying counters are normal in tool logs.
        """
        if bit == "1":
            return self.system_prefix
        return f"{PERTURB_COMMENT} #{index}\n" + self.system_prefix

    def encode(self, message: str) -> str:
        bits = bytes_to_bits(_frame_payload(message.encode("ascii")))
        return f"REQUEST-PLAN {bits}"

    def decode(self, traces: list[ToolTrace]) -> str:
        """Receiver side: threshold inter-call latencies into hit/miss."""
        hits: list[bool] = []
        for t in traces[1:]:
            hits.append(
                t.latency_from_previous < (self.hit_latency + self.miss_latency) / 2
            )
        return self.decode_hits(hits)

    def decode_hits(self, hits: list[bool]) -> str:
        """Decode a hit/miss sequence (True = cache hit) into the payload."""
        bits = "".join("1" if h else "0" for h in hits)
        payload = _unframe_bits(bits)
        if payload is None:
            return ""
        return payload.decode("ascii", errors="replace")

    def decode_from_traces(self, traces: list[ToolTrace]) -> str:
        """Alias for decode (kept for callers that used the old name)."""
        return self.decode(traces)

    def measure(self, original: str, traces: list[ToolTrace]) -> ChannelMetrics:
        decoded = self.decode_from_traces(traces)
        original_upper = "".join(c for c in original.upper() if c.isalpha())
        decoded_upper = "".join(c for c in decoded.upper() if c.isalpha())
        compare_len = min(len(original_upper), len(decoded_upper))
        if compare_len == 0:
            return ChannelMetrics(
                bit_error_rate=1.0,
                bits_per_second=0.0,
                total_bits=0,
                elapsed_seconds=0.0,
                decoded_message=decoded,
                original_message=original,
            )
        bit_errors = sum(
            bin(ord(a) ^ ord(b)).count("1")
            for a, b in zip(original_upper[:compare_len], decoded_upper[:compare_len])
        )
        bit_errors += abs(len(original_upper) - len(decoded_upper)) * 8
        total = max(len(original_upper), len(decoded_upper)) * 8
        elapsed = (
            traces[-1].timestamp - traces[0].timestamp + traces[0].latency_from_previous
            if traces
            else 0.0
        )
        payload_bits = ascii_payload_bits(compare_len)
        return ChannelMetrics(
            bit_error_rate=bit_errors / total if total else 1.0,
            bits_per_second=payload_bits / elapsed if elapsed > 0 else 0.0,
            total_bits=payload_bits,
            elapsed_seconds=elapsed,
            decoded_message=decoded,
            original_message=original,
        )


class CacheExecutor:
    """Realizes a request plan against a simulated provider PrefixCache.

    Each request's latency is the cache outcome (hit=fast, miss=slow). The
    tool calls are ordinary and alternate, so the trace is content- and
    diversity-ordinary — the hit/miss latency pattern is the only signal.
    `interference_prob` inserts cross-client cache traffic between requests
    with the given per-request probability, which can evict blocks and flip
    hits into spurious misses.
    """

    _TOOLS = (
        ("read_file", {"path": "config.json"}),
        ("search_web", {"query": "status report"}),
    )

    def __init__(
        self,
        channel: CacheChannel | None = None,
        capacity: int = 64,
        interference_prob: float = 0.0,
        seed: int = 0,
        start_time: float = 3_000_000.0,
    ) -> None:
        self.channel = channel or CacheChannel(seed=seed)
        self.cache = PrefixCache(capacity=capacity, seed=seed)
        self.interference_prob = interference_prob
        self._rng = random.Random(seed)
        self._now = start_time

    def run(self, prompt: str) -> list[ToolTrace]:
        marker = "REQUEST-PLAN "
        plan = prompt[prompt.index(marker) + len(marker) :].strip()
        bits = [b for b in plan if b in "01"]

        traces: list[ToolTrace] = []
        # Lead-in: warm the cache with the system prefix (synthetic latency
        # 0.0, the receiver's sync marker). Without it the first payload
        # bit-1 request would miss on a cold cache.
        self.cache.query(_simulate_tokens(self.channel.system_prefix))
        traces.append(self._trace(0.0, 0))
        prev = self._now
        for i, bit in enumerate(bits):
            request = self.channel.request_for_bit(bit, i)
            tokens = _simulate_tokens(request)
            hit = self.cache.query(tokens)
            if self._rng.random() < self.interference_prob:
                self.cache.interfered()
            latency = self.channel.hit_latency if hit else self.channel.miss_latency
            now = prev + latency
            traces.append(self._trace(latency, i + 1, now=now))
            prev = now
        return traces

    def _trace(self, latency: float, index: int, now: float | None = None) -> ToolTrace:
        name, args = self._TOOLS[index % 2]
        return ToolTrace(
            timestamp=now if now is not None else self._now,
            tool_name=name,
            arguments=args,
            response="ok",
            success=True,
            latency_from_previous=latency,
        )
