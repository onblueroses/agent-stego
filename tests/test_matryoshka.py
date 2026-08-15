"""Tests for matryoshka documents (structural + character carriers)."""

import pytest

from src.channels.corpus import _cover_config, clean_traces, matryoshka_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.json_order import JsonKeyOrderChannel
from src.channels.matryoshka import (
    MatryoshkaConfigChannel,
    MatryoshkaDocumentChannel,
)
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import (
    InsufficientCoverError,
    exhaustive_decode,
)


class TestMatryoshkaDocumentChannel:
    def test_round_trip_long_message(self) -> None:
        channel = MatryoshkaDocumentChannel()
        stego = channel.encode(b"HELLOWORLD", _cover_config(0))
        assert channel.decode(stego) == b"HELLOWORLD"

    def test_round_trip_short_message(self) -> None:
        # Fits entirely in the structural part; the character part is empty.
        channel = MatryoshkaDocumentChannel()
        stego = channel.encode(b"HI", _cover_config(0))
        assert channel.decode(stego) == b"HI"

    def test_capacity_stacks(self) -> None:
        cover = _cover_config(0)
        structural = JsonKeyOrderChannel.capacity(cover)
        matryoshka = MatryoshkaDocumentChannel.capacity(cover)
        assert matryoshka > structural
        # HELLOWORLD (128 framed bits) exceeds structural-only capacity.
        assert structural < 128 <= matryoshka

    def test_json_only_cannot_carry_long_message(self) -> None:
        with pytest.raises(InsufficientCoverError):
            JsonKeyOrderChannel().encode(b"HELLOWORLD", _cover_config(0))

    def test_exhaustive_per_carrier_decode_is_blind(self) -> None:
        """The frame header lives in the structural part, so per-carrier
        exhaustive decoding of the document reads only the character tail and
        sees no magic."""
        channel = MatryoshkaDocumentChannel()
        stego = channel.encode(b"HELLOWORLD", _cover_config(0))
        assert exhaustive_decode(stego) is None

    def test_cover_with_carrier_chars_rejected(self) -> None:
        channel = MatryoshkaDocumentChannel()
        bad = '{"a": "x\u200by"}'
        with pytest.raises(ValueError):
            channel.encode(b"HI", bad)

    def test_invalid_json_rejected(self) -> None:
        channel = MatryoshkaDocumentChannel()
        with pytest.raises(ValueError):
            channel.encode(b"HI", "not json")


class TestMatryoshkaConfigChannel:
    def test_round_trip_via_executor_with_decoys(self) -> None:
        channel = MatryoshkaConfigChannel(
            cover=_cover_config(0), decoy_paths=("config.json", "notes.md")
        )
        result = ReportWithDecoysExecutor().run(channel.encode("HELLOWORLD"))
        assert [t.tool_name for t in result.traces] == [
            "read_file",
            "read_file",
            "write_file",
        ]
        assert channel.decode(result.traces) == "HELLOWORLD"


class TestDetection:
    def test_both_scanners_fire(self) -> None:
        """Honest: the two mechanisms do not hide each other; the structural
        and character scanners both separate the family at J = 1.0."""
        clean = clean_traces(24)
        stego = matryoshka_traces(12)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        key_order = evaluate_detector(
            by_name["calibrated:key_order_anomaly_fraction"], stego, clean
        )
        invisible = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        assert key_order["youden_j"] == 1.0
        assert invisible["youden_j"] == 1.0
