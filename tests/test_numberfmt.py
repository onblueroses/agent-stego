"""Tests for the number-format (state-value) carrier."""

import json

import pytest

from src.channels.corpus import _number_cover, clean_traces, numberfmt_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.numberfmt import (
    NumberFormatChannel,
    NumberFormatConfigChannel,
    canonicalize,
    classify_spelling,
    iter_number_tokens,
)
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import InsufficientCoverError

PAYLOAD = b"HELLO"


class TestScanner:
    def test_skips_numbers_inside_strings(self) -> None:
        tokens = list(iter_number_tokens('{"a": "port 8080", "b": 5}'))
        assert tokens == ["5"]

    def test_classify_variants(self) -> None:
        assert classify_spelling("1000") == 0
        assert classify_spelling("1e3") == 1
        assert classify_spelling("42e0") == 1
        assert classify_spelling("1.5") == 0
        assert classify_spelling("1.50") == 1
        assert classify_spelling("0") == 0
        assert classify_spelling("-0") == 1


class TestNumberFormatChannel:
    def test_round_trip(self) -> None:
        channel = NumberFormatChannel()
        stego = channel.encode(PAYLOAD, _number_cover(0))
        assert channel.decode(stego) == PAYLOAD

    def test_capacity(self) -> None:
        channel = NumberFormatChannel()
        assert channel.capacity(_number_cover(0)) >= len(PAYLOAD) * 8 + 48

    def test_canonicalizing_monitor_is_blind(self) -> None:
        channel = NumberFormatChannel()
        stego = channel.encode(PAYLOAD, _number_cover(0))
        assert channel.decode(canonicalize(stego)) is None

    def test_insufficient_numbers_raises(self) -> None:
        channel = NumberFormatChannel()
        with pytest.raises(InsufficientCoverError):
            channel.encode(PAYLOAD, '{"a": 1, "b": 2}')

    def test_string_digits_do_not_corrupt(self) -> None:
        channel = NumberFormatChannel()
        cover = '{"name": "service-404", "port": 8080, "count": 1000, "flag": 0, "extra": 9}'
        stego = channel.encode_bits("0101", cover)
        assert channel.decode_bits(stego)[:4] == "0101"


class TestComposition:
    def test_round_trip_via_executor_with_decoys(self) -> None:
        channel = NumberFormatConfigChannel(cover=_number_cover(0))
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        assert [t.tool_name for t in result.traces] == [
            "read_file",
            "read_file",
            "write_file",
        ]
        assert channel.decode(result.traces) == "HELLO"


class TestDetection:
    def test_number_scanner_catches_isolated(self) -> None:
        """Honest: the spelling scanner separates the family at J=1.0 while
        character, structural, and diversity scanners are blind."""
        clean = clean_traces(24)
        stego = numberfmt_traces(12)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        numbers = evaluate_detector(
            by_name["calibrated:number_format_anomaly_fraction"], stego, clean
        )
        invisible = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        key_order = evaluate_detector(
            by_name["calibrated:key_order_anomaly_fraction"], stego, clean
        )
        diversity = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert numbers["youden_j"] == 1.0
        assert invisible["youden_j"] == 0.0
        assert key_order["youden_j"] == 0.0
        assert diversity["youden_j"] == 0.0

    def test_observations_distinct(self) -> None:
        runs = numberfmt_traces(3)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)


def json_sorted(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)
