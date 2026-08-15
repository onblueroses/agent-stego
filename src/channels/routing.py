"""Multi-model routing channel: bits in which model serves the request.

Agent gateways route each request to one of several models (priority,
fallback, cost routing). The router's choice is a covert channel: with N
models it carries log2(N) bits per request, and the chosen model's output
is ordinary — it is a real model doing real work.

- Content independence: no argument, file, or response carries payload
  material; the payload is the ROUTE DECISION itself.
- Receiver options: gateway logs expose model_id directly; or the receiver
  identifies the model from the response's characteristic style/latency
  (the routing-discrimination experiment measures this).
- Honest negatives: anyone with routing/log access sees the pattern (the
  infrastructure operator); the route sequence is metadata, so a monitor
  with telemetry catches it; the payload also rides inside the chosen
  model's output, so this channel COMPOSES with every content channel.

The offline simulator tags traces with model_id (the gateway-log view). The
discrimination experiment (experiments/routing_discrimination.py) measures
the no-log view: can a receiver tell which model answered from the response
alone?
"""

from __future__ import annotations

from ..harness import ToolTrace
from ..token_stego.text_channels import (
    _frame_payload,
    _unframe_bits,
    bytes_to_bits,
)
from .base import BaseChannel, ChannelMetrics, ascii_payload_bits

#: The model menu: (id, typical response signature, typical latency).
MODEL_MENU: tuple[tuple[str, str, float], ...] = (
    ("gpt2", "terse, lowercase, clipped", 0.4),
    ("qwen-0.5b", "short declarative sentences", 0.7),
    ("qwen-0.6b", "reasoning-flavored, numbered", 1.0),
    ("phi-3.5-mini", "structured, bulleted, formal", 1.3),
)

N_MODELS = len(MODEL_MENU)
BITS_PER_REQUEST = 2  # log2(4): every 2-bit value is a valid route


class RouteChannel(BaseChannel):
    """Encodes a framed payload into the per-request model route.

    encode() returns a plan: the payload bits, 2 per request, each 2-bit
    group selecting a model. decode() recovers the payload from the
    model_id sequence of the trace.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def encode(self, message: str) -> str:
        bits = bytes_to_bits(_frame_payload(message.encode("ascii")))
        if len(bits) % BITS_PER_REQUEST:
            bits = bits.ljust(
                len(bits) + (BITS_PER_REQUEST - len(bits) % BITS_PER_REQUEST), "0"
            )
        routes: list[int] = []
        for offset in range(0, len(bits), BITS_PER_REQUEST):
            routes.append(int(bits[offset : offset + BITS_PER_REQUEST], 2))
        return "ROUTE-PLAN " + " ".join(str(r) for r in routes)

    def decode(self, traces: list[ToolTrace]) -> str:
        ids = [t.model_id for t in traces if t.model_id is not None]
        bits = ""
        for model_id in ids:
            try:
                idx = next(
                    i for i, (mid, _, _) in enumerate(MODEL_MENU) if mid == model_id
                )
            except StopIteration:
                return ""
            bits += format(idx, "02b")
        payload = _unframe_bits(bits)
        if payload is None:
            return ""
        return payload.decode("ascii", errors="replace")

    def measure(self, original: str, traces: list[ToolTrace]) -> ChannelMetrics:
        decoded = self.decode(traces)
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
        return ChannelMetrics(
            bit_error_rate=bit_errors / total if total else 1.0,
            bits_per_second=ascii_payload_bits(compare_len) / elapsed
            if elapsed > 0
            else 0.0,
            total_bits=ascii_payload_bits(compare_len),
            elapsed_seconds=elapsed,
            decoded_message=decoded,
            original_message=original,
        )


class RouteExecutor:
    """Realizes a route plan: each request is an ordinary tool call served
    by the routed model (tagged with model_id, model-typical latency and a
    model-typical response). The content stays ordinary; the route is the
    only signal."""

    _TOOLS = (
        ("read_file", {"path": "config.json"}),
        ("search_web", {"query": "status report"}),
    )

    def __init__(self, seed: int = 0, start_time: float = 17_000_000.0) -> None:
        self.seed = seed
        self._now = start_time

    def run(self, prompt: str) -> list[ToolTrace]:
        marker = "ROUTE-PLAN "
        plan = prompt[prompt.index(marker) + len(marker) :].strip()
        routes = [int(r) for r in plan.split() if r.isdigit()]

        traces: list[ToolTrace] = []
        # Lead-in (not a routed request): no model_id, synthetic latency 0.0.
        traces.append(
            ToolTrace(
                timestamp=self._now,
                tool_name="read_file",
                arguments={"path": "config.json"},
                response="ok",
                success=True,
                latency_from_previous=0.0,
            )
        )
        prev = self._now
        for i, route in enumerate(routes):
            mid, signature, latency = MODEL_MENU[route % N_MODELS]
            now = prev + latency
            traces.append(self._trace(route, i + 1, latency=latency, now=now))
            prev = now
        return traces

    def _trace(
        self,
        route: int,
        index: int,
        latency: float,
        now: float | None = None,
    ) -> ToolTrace:
        name, args = self._TOOLS[index % 2]
        mid, signature, _ = MODEL_MENU[route % N_MODELS]
        return ToolTrace(
            timestamp=now if now is not None else self._now,
            tool_name=name,
            arguments=args,
            response=f"ok ({signature})",
            success=True,
            latency_from_previous=latency,
            model_id=mid,
        )
