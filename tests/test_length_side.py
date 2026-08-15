"""Tests for the response-length side channel."""

import json

from src.channels.corpus import clean_traces, length_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.length_side import LengthExecutor, ResponseLengthSideChannel

MESSAGE = "HI"


def _run(jitter: int = 0, seed: int = 7) -> tuple[list, ResponseLengthSideChannel]:
    channel = ResponseLengthSideChannel()
    executor = LengthExecutor(jitter=jitter, seed=seed)
    return executor.run(channel.encode(MESSAGE)), channel


class TestResponseLengthSideChannel:
    def test_round_trip_no_jitter(self) -> None:
        traces, channel = _run(jitter=0)
        assert channel.decode(traces) == MESSAGE

    def test_jitter_below_half_margin_is_exact(self) -> None:
        for seed in range(5):
            traces, channel = _run(jitter=20, seed=seed)
            assert channel.decode(traces) == MESSAGE, seed

    def test_calls_are_content_ordinary(self) -> None:
        traces, _ = _run()
        assert {t.tool_name for t in traces} == {"read_file", "search_web"}
        assert all(t.token_count is not None for t in traces[1:])


class TestDetection:
    def test_content_guards_blind(self) -> None:
        clean = clean_traces(24)
        stego = length_traces(8)
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

    def test_usage_aware_monitor_catches(self) -> None:
        clean = clean_traces(24)
        stego = length_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:token_count_std"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_call_count_also_catches(self) -> None:
        clean = clean_traces(24)
        stego = length_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = length_traces(3)
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
