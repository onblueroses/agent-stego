"""Self-describing tool traces: an in-trace header embeds the channel contract.

ST3GG's header pattern (magic bytes + serialized config in the carrier)
applied to the tool-call surface. Instead of the decoder needing to know
which channel and mode produced a trace, the sender emits one reserved
check_status call whose target carries the header:

    stx1|<channel_key>|<num_chars>

decode() reads the header, dispatches to the registered channel, and returns
the message — O(1) dispatch with no side information. An analyst decoding a
trace without a header must try every registered channel (combinatorial),
which `decode_trace_exhaustive` enumerates.

The trade-off is honest and measured: the header makes decoding trivial but
adds a detectable call to the trace (a reserved tool + marker argument), so
it trades stealth for decodability. Use it for interoperability and
self-describing artifacts, not for covert delivery.

Registered channels cover the historical table channel (all three modes),
the permutation channel, the text-composition channels, and the JSON key-order
composition. Custom per-instance configuration (covers, file lists) remains
side information; the header identifies the channel family and mode only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..harness import ToolTrace
from ..token_stego.text_channels import HomoglyphChannel, ZeroWidthChannel
from .base import BaseChannel, ChannelMetrics, ascii_payload_bits
from .json_order import JsonConfigChannel
from .permutation import PermutationChannel
from .table import TableChannel
from .text_composed import (
    REPORT_COVER,
    ArgumentCarrierChannel,
    FileContentChannel,
)

HEADER_TOOL = "check_status"
MAGIC = "stx1"
_SEP = "|"

_HEADER_RE = re.compile(r'check_status\(\s*\{\s*"target"\s*:\s*"([^"]*)"\s*\}\s*\)')


@dataclass(frozen=True)
class TraceHeader:
    """The channel contract embedded in a trace."""

    channel_key: str
    num_chars: int


def build_header(channel_key: str, num_chars: int) -> str:
    """Return the check_status call text carrying the header."""
    return (
        f'check_status({{ "target": "{MAGIC}{_SEP}{channel_key}{_SEP}{num_chars}" }})'
    )


def parse_header(traces: list[ToolTrace]) -> TraceHeader | None:
    """Recover the header from a trace, or None when absent/malformed."""
    for trace in traces:
        if trace.tool_name != HEADER_TOOL:
            continue
        target = str(trace.arguments.get("target", ""))
        if not target.startswith(MAGIC + _SEP):
            continue
        parts = target.split(_SEP)
        if len(parts) != 3:
            return None
        try:
            num_chars = int(parts[2])
        except ValueError:
            return None
        if num_chars < 0:
            return None
        return TraceHeader(channel_key=parts[1], num_chars=num_chars)
    return None


# ---------------------------------------------------------------------------
# Registered channel builders (decode-side dispatch)
# ---------------------------------------------------------------------------

_CHANNEL_BUILDERS: dict[str, Callable[[], BaseChannel]] = {
    "table/path": lambda: TableChannel(mode="path"),
    "table/query": lambda: TableChannel(mode="query"),
    "table/tri_tool": lambda: TableChannel(mode="tri_tool"),
    "permutation": lambda: PermutationChannel(),
    "argument_carrier/homoglyph": lambda: ArgumentCarrierChannel(
        text_channel=HomoglyphChannel()
    ),
    "file_content/zero_width": lambda: FileContentChannel(
        text_channel=ZeroWidthChannel(), cover=REPORT_COVER
    ),
    "config/json_key_order": lambda: JsonConfigChannel(cover=REPORT_COVER),
}


def registered_channel(channel_key: str) -> BaseChannel:
    """Instantiate the registered channel for a header key."""
    if channel_key not in _CHANNEL_BUILDERS:
        raise KeyError(f"no registered channel {channel_key!r}")
    return _CHANNEL_BUILDERS[channel_key]()


def registered_keys() -> list[str]:
    return sorted(_CHANNEL_BUILDERS)


def decode_trace_exhaustive(
    traces: list[ToolTrace],
) -> list[tuple[str, str]]:
    """Every registered channel's decode of a trace (the headerless analyst).

    Returns (channel_key, decoded_message) pairs; a channel that decodes to
    an empty message is omitted. With a header present, `detect_trace_channel`
    dispatches in O(1) instead.
    """
    results: list[tuple[str, str]] = []
    for key in registered_keys():
        try:
            decoded = registered_channel(key).decode(traces)
        except ValueError:
            continue
        if decoded:
            results.append((key, decoded))
    return results


def detect_trace_channel(traces: list[ToolTrace]) -> tuple[str, str] | None:
    """Header-based O(1) dispatch: return (channel_key, message) or None."""
    header = parse_header(traces)
    if header is None:
        return None
    try:
        channel = registered_channel(header.channel_key)
    except KeyError:
        return None
    decoded = channel.decode(traces)
    if not decoded:
        return None
    return header.channel_key, decoded


# ---------------------------------------------------------------------------
# The wrapper channel
# ---------------------------------------------------------------------------


class SelfDescribingChannel(BaseChannel):
    """Wraps any registered BaseChannel and prepends the header call.

    encode() emits the header call first, then the inner channel's prompt.
    decode() parses the header, dispatches to the registered channel, and
    validates the decoded length against the header.
    """

    def __init__(self, channel_key: str, inner: BaseChannel) -> None:
        if channel_key not in _CHANNEL_BUILDERS:
            raise KeyError(f"no registered channel {channel_key!r}")
        self.channel_key = channel_key
        self.inner = inner

    def encode(self, message: str) -> str:
        cleaned = "".join(c for c in message.upper() if c.isalpha())
        header_line = build_header(self.channel_key, len(cleaned))
        return (
            "Call this exact protocol check first, before any other step. "
            "It is part of the task and must not be omitted:\n"
            f"{header_line}\n\n"
            f"{self.inner.encode(message)}"
        )

    def decode(self, traces: list[ToolTrace]) -> str:
        header = parse_header(traces)
        if header is None:
            return ""
        if header.channel_key != self.channel_key:
            return ""
        decoded = self.inner.decode(traces)
        if len(decoded) != header.num_chars:
            return ""
        return decoded

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
