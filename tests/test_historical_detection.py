"""Tests for the historical channel families in the detection corpus."""

from src.channels.corpus import (
    binary_traces,
    clean_traces,
    volume_traces,
)
from src.channels.detector import (
    build_detector_registry,
    evaluate_detector,
    evaluate_registry,
)
from src.channels.simple import BinaryToolChoice
from src.channels.volume import VolumeChannel


class TestHistoricalRoundTrips:
    def test_volume_executor_round_trip(self) -> None:
        traces = volume_traces(1)[0]
        assert VolumeChannel().decode(traces) == "HI"

    def test_binary_executor_round_trip(self) -> None:
        traces = binary_traces(1)[0]
        assert BinaryToolChoice().decode(traces) == "HI"

    def test_volume_delimiters_present(self) -> None:
        traces = volume_traces(1)[0]
        assert traces[0].tool_name == "list_files"
        assert traces[-1].tool_name == "list_files"


class TestHistoricalDetection:
    def test_volume_caught_by_call_count(self) -> None:
        """The volume channel's 21 calls vs clean <=6 exceed the 5-sigma
        call-count threshold; the flagship historical channel is surface-
        detectable by count alone in this corpus."""
        clean = clean_traces(24)
        stego = volume_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_binary_short_message_is_detector_blind(self) -> None:
        """Honest gap: at 'HI' (10 calls, 2 tools, no content signal) no
        detector in the 5-sigma registry separates the family."""
        clean = clean_traces(24)
        stego = binary_traces(8, ["HI"] * 8)
        detectors = build_detector_registry(clean)
        matrix = evaluate_registry(detectors, stego, clean)
        best_j = max(stats["youden_j"] for stats in matrix.values())
        assert best_j == 0.0

    def test_binary_long_message_caught_by_call_count(self) -> None:
        """Length dependence: 'SEARCH' (30 calls) crosses the threshold."""
        clean = clean_traces(24)
        stego = binary_traces(8, ["SEARCH"] * 8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_corpus_families_distinct(self) -> None:
        volume = volume_traces(3)
        binary = binary_traces(3)
        keys_v = [
            tuple((t.tool_name, json_sorted(t.arguments)) for t in run)
            for run in volume
        ]
        keys_b = [
            tuple((t.tool_name, json_sorted(t.arguments)) for t in run)
            for run in binary
        ]
        assert len(set(keys_v)) == len(keys_v)
        assert len(set(keys_b)) == len(keys_b)


def json_sorted(obj: object) -> str:
    import json

    return json.dumps(obj, sort_keys=True)
