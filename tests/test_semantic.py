"""Tests for the offline-simulated semantic (synonym-choice) carrier."""

import json

import pytest

from src.channels.corpus import clean_traces, semantic_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.semantic import (
    SemanticReportChannel,
    SynonymChoiceChannel,
    _anchor_cover,
    canonicalize,
)
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import InsufficientCoverError, exhaustive_decode

PAYLOAD = b"HELLO"


class TestSynonymChoiceChannel:
    def test_round_trip(self) -> None:
        channel = SynonymChoiceChannel()
        stego = channel.encode(PAYLOAD, _anchor_cover(120))
        assert channel.decode(stego) == PAYLOAD

    def test_capacity(self) -> None:
        channel = SynonymChoiceChannel()
        assert channel.capacity(_anchor_cover(120)) >= len(PAYLOAD) * 8 + 48

    def test_canonicalizing_monitor_is_blind(self) -> None:
        channel = SynonymChoiceChannel()
        stego = channel.encode(PAYLOAD, _anchor_cover(120))
        assert channel.decode(canonicalize(stego)) is None

    def test_character_exhaustive_decoder_is_blind(self) -> None:
        """The semantic carrier is invisible to the character-channel
        exhaustive decoder (distinct taxonomy axis)."""
        channel = SynonymChoiceChannel()
        stego = channel.encode(PAYLOAD, _anchor_cover(120))
        assert exhaustive_decode(stego) is None

    def test_non_canonical_cover_rejected(self) -> None:
        channel = SynonymChoiceChannel()
        with pytest.raises(ValueError):
            channel.encode_bits("00", "The team was quite good today.")

    def test_insufficient_capacity_raises(self) -> None:
        channel = SynonymChoiceChannel()
        with pytest.raises(InsufficientCoverError):
            channel.encode(PAYLOAD, "The very good team.")


class TestSemanticReportChannel:
    def test_round_trip_via_executor_with_decoys(self) -> None:
        channel = SemanticReportChannel(cover=_anchor_cover(120))
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        assert [t.tool_name for t in result.traces] == [
            "read_file",
            "read_file",
            "write_file",
        ]
        assert channel.decode(result.traces) == "HELLO"


class TestDetection:
    def test_synonym_scanner_catches_isolated(self) -> None:
        """Honest: the synonym-surface scanner separates the family at J=1.0
        while the character and diversity scanners are blind — the semantic
        axis is orthogonal."""
        clean = clean_traces(24)
        stego = semantic_traces(12)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        synonym = evaluate_detector(
            by_name["calibrated:synonym_surface_fraction"], stego, clean
        )
        invisible = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        diversity = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert synonym["youden_j"] == 1.0
        assert invisible["youden_j"] == 0.0
        assert diversity["youden_j"] == 0.0

    def test_observations_distinct(self) -> None:
        runs = semantic_traces(3)
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
