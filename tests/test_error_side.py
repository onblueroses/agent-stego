"""Tests for the error-pattern side channel."""

import json

from src.channels.corpus import clean_traces, error_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.error_side import ErrorSideChannel, SuccessExecutor

MESSAGE = "HI"


def _run() -> tuple[list, ErrorSideChannel]:
    channel = ErrorSideChannel()
    executor = SuccessExecutor()
    return executor.run(channel.encode(MESSAGE)), channel


class TestErrorSideChannel:
    def test_round_trip(self) -> None:
        traces, channel = _run()
        assert channel.decode(traces) == MESSAGE

    def test_failed_calls_carry_ones(self) -> None:
        traces, _ = _run()
        # Frame bits contain both zeros and ones; failures must exist.
        assert any(not t.success for t in traces[1:])

    def test_calls_are_content_ordinary(self) -> None:
        traces, _ = _run()
        assert {t.tool_name for t in traces} == {"read_file", "search_web"}


class TestDetection:
    def test_content_guards_blind(self) -> None:
        clean = clean_traces(24)
        stego = error_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        content_features = [
            "invisible_char_density",
            "nbsp_density",
            "homoglyph_density",
            "combining_mark_density",
            "non_ascii_density",
            "key_order_anomaly_fraction",
            "duplicate_key_fraction",
            "synonym_surface_fraction",
            "number_format_anomaly_fraction",
        ]
        for feature in content_features:
            stats = evaluate_detector(by_name[f"calibrated:{feature}"], stego, clean)
            assert stats["youden_j"] == 0.0, feature

    def test_reliability_monitor_catches(self) -> None:
        clean = clean_traces(24)
        stego = error_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:error_fraction"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = error_traces(3)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 3))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)


def json_sorted(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)
