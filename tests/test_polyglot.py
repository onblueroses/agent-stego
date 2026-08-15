"""Tests for polyglot documents (JSON duplicate-key parser differential)."""

import json

import pytest

from src.channels.corpus import _polyglot_cover, clean_traces, polyglot_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.polyglot import (
    DuplicateKeyChannel,
    DuplicateKeyConfigChannel,
    first_wins,
    parse_preserving,
)
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import InsufficientCoverError

PAYLOAD = b"HELLO"


class TestParser:
    def test_preserving_parser_keeps_duplicates(self) -> None:
        pairs = parse_preserving('{"a": 1, "a": 2, "b": 3}')
        assert pairs is not None
        assert pairs == [("a", 1), ("a", 2), ("b", 3)]

    def test_preserving_parser_nested_values(self) -> None:
        pairs = parse_preserving('{"a": {"x": [1, 2]}, "a": 0}')
        assert pairs is not None
        assert pairs[0][1] == {"x": [1, 2]}
        assert pairs[1] == ("a", 0)

    def test_invalid_object_is_none(self) -> None:
        assert parse_preserving("not json") is None
        assert parse_preserving("[1, 2]") is None

    def test_first_wins_collapses_duplicates(self) -> None:
        out = first_wins('{"a": 1, "a": 2}')
        assert json.loads(out) == {"a": 1}


class TestDuplicateKeyChannel:
    def test_round_trip(self) -> None:
        channel = DuplicateKeyChannel()
        stego = channel.encode(PAYLOAD, _polyglot_cover(0))
        assert channel.decode(stego) == PAYLOAD

    def test_first_wins_monitor_is_blind(self) -> None:
        channel = DuplicateKeyChannel()
        stego = channel.encode(PAYLOAD, _polyglot_cover(0))
        assert channel.decode(first_wins(stego)) is None

    def test_any_reserialization_collapses(self) -> None:
        """Both win-orders destroy the channel once re-serialized, because the
        duplicate structure itself is gone — the axis is parser/canonicalizer
        behavior, not which duplicate value wins."""
        channel = DuplicateKeyChannel()
        stego = channel.encode(PAYLOAD, _polyglot_cover(0))
        last_wins_text = json.dumps(json.loads(stego), indent=2)  # Python last-wins
        assert channel.decode(last_wins_text) is None

    def test_capacity_is_key_count(self) -> None:
        channel = DuplicateKeyChannel()
        assert channel.capacity(_polyglot_cover(0)) >= len(PAYLOAD) * 8 + 48

    def test_insufficient_keys_raises(self) -> None:
        channel = DuplicateKeyChannel()
        with pytest.raises(InsufficientCoverError):
            channel.encode(PAYLOAD, '{"a": 1, "b": 2}')

    def test_corrupt_bit_value_fails_closed(self) -> None:
        channel = DuplicateKeyChannel()
        stego = channel.encode(PAYLOAD, _polyglot_cover(0))
        # Corrupt the duplicated (bit) value of "environment" — its cover
        # occurrence is a string, so only the bit occurrence matches 0.
        bad = stego.replace('"environment": 0', '"environment": "x"')
        assert bad != stego
        assert channel.decode(bad) is None


class TestComposition:
    def test_round_trip_via_executor_with_decoys(self) -> None:
        channel = DuplicateKeyConfigChannel(
            cover=_polyglot_cover(0), decoy_paths=("config.json", "notes.md")
        )
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        assert [t.tool_name for t in result.traces] == [
            "read_file",
            "read_file",
            "write_file",
        ]
        assert channel.decode(result.traces) == "HELLO"


class TestDetection:
    def test_duplicate_scanner_catches(self) -> None:
        """Honest: the duplicate structure is visible to a duplicate-aware
        scanner at J = 1.0; the parser differential protects only against
        canonicalizing monitors."""
        clean = clean_traces(24)
        stego = polyglot_traces(12)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:duplicate_key_fraction"], stego, clean
        )
        assert stats["youden_j"] == 1.0

    def test_diversity_blind_with_decoys(self) -> None:
        clean = clean_traces(24)
        stego = polyglot_traces(12)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert stats["youden_j"] == 0.0

    def test_observations_distinct(self) -> None:
        runs = polyglot_traces(3)
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
