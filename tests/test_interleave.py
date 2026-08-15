"""Tests for the concurrency-interleaving channel."""

from src.channels.corpus import clean_traces, interleave_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.interleave import InterleaveChannel, InterleaveExecutor

MESSAGE = "HELLO"


class TestInterleaveChannel:
    def test_encode_decode_round_trip(self) -> None:
        channel = InterleaveChannel()
        executor = InterleaveExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        assert channel.decode(traces) == MESSAGE

    def test_round_trips_many_seeds(self) -> None:
        for trial in range(5):
            channel = InterleaveChannel(seed=14_000 + trial)
            executor = InterleaveExecutor(seed=14_000 + trial)
            traces = executor.run(channel.encode(MESSAGE))
            assert channel.decode(traces) == MESSAGE

    def test_all_calls_are_ordinary(self) -> None:
        channel = InterleaveChannel()
        executor = InterleaveExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        for t in traces[1:]:
            assert t.tool_name in ("read_file", "search_web")
            assert MESSAGE not in str(t.arguments)


class TestDetection:
    def test_registry_has_no_order_feature(self) -> None:
        """Honest positive: no corpus-detector feature models intra-pair
        call order, and timing is uniform, so latency/content features stay
        at chance; the honest catch is volume."""
        clean = clean_traces(24)
        stego = interleave_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name in (
            "calibrated:invisible_char_density",
            "calibrated:homoglyph_density",
            "calibrated:non_ascii_density",
            "calibrated:mean_latency",
            "calibrated:latency_std",
            "calibrated:unique_tools",
        ):
            res = evaluate_detector(by_name[name], stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"

    def test_volume_signature_fires(self) -> None:
        """Honest negative: 2 calls per bit is a strong volume signature."""
        clean = clean_traces(24)
        stego = interleave_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        res = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert res["youden_j"] >= 0.8

    def test_observations_distinct(self) -> None:
        runs = interleave_traces(3)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
