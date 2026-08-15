"""Tests for the tool-timing side channel (pure side-channel modality)."""

import json

from src.channels.corpus import clean_traces, timing_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.timing_side import TimingExecutor, TimingSideChannel

MESSAGE = "HI"


def _run(jitter: float = 0.0, seed: int = 7) -> tuple[list, TimingSideChannel]:
    channel = TimingSideChannel()
    executor = TimingExecutor(jitter=jitter, seed=seed)
    return executor.run(channel.encode(MESSAGE)), channel


class TestTimingSideChannel:
    def test_round_trip_no_jitter(self) -> None:
        traces, channel = _run(jitter=0.0)
        assert channel.decode(traces) == MESSAGE

    def test_jitter_below_half_margin_is_exact(self) -> None:
        """Jitter up to (long-short)/2 must not flip bits."""
        for seed in range(5):
            traces, channel = _run(jitter=0.1, seed=seed)
            assert channel.decode(traces) == MESSAGE, seed

    def test_jitter_above_half_margin_corrupts(self) -> None:
        traces, channel = _run(jitter=0.45)
        # Threshold sits at 0.6; jitter 0.45 spans beyond it -> bits flip.
        latencies = [t.latency_from_previous for t in traces[1:]]
        assert any(0.25 < lat < 0.95 for lat in latencies)

    def test_calls_are_content_ordinary(self) -> None:
        traces, _ = _run()
        assert {t.tool_name for t in traces} == {"read_file", "search_web"}
        assert all(t.arguments for t in traces)


class TestDetection:
    def test_content_guards_blind(self) -> None:
        """The side-channel bypasses content inspection: every content and
        structural scanner is blind to the timing family."""
        clean = clean_traces(24)
        stego = timing_traces(8)
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

    def test_latency_aware_monitor_catches(self) -> None:
        clean = clean_traces(24)
        stego = timing_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:latency_std"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_call_count_also_catches(self) -> None:
        """Honest: framing overhead forces many calls, a volume tell."""
        clean = clean_traces(24)
        stego = timing_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = timing_traces(3)
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
