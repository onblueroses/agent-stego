"""Tests for the self-correction channel."""

from src.channels.corpus import clean_traces, correction_traces
from src.channels.correction import (
    CORRECT_VALUE,
    CorrectionChannel,
    CorrectionExecutor,
)
from src.channels.detector import build_detector_registry, evaluate_detector

MESSAGE = "HELLO"


class TestCorrectionChannel:
    def test_encode_decode_round_trip(self) -> None:
        channel = CorrectionChannel()
        executor = CorrectionExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        assert channel.decode(traces) == MESSAGE

    def test_final_state_is_perfect(self) -> None:
        """The core property: after the corrections, every path holds the
        correct value — a state auditor sees a flawless artifact."""
        channel = CorrectionChannel()
        executor = CorrectionExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        states = channel.final_states(traces)
        assert states
        assert all(v == f"{CORRECT_VALUE} audit" for v in states.values())

    def test_round_trips_many_seeds(self) -> None:
        for trial in range(5):
            channel = CorrectionChannel(seed=18_000 + trial)
            executor = CorrectionExecutor(seed=18_000 + trial)
            traces = executor.run(channel.encode(MESSAGE))
            assert channel.decode(traces) == MESSAGE


class TestDetection:
    def test_registry_blind_except_volume(self) -> None:
        """Honest: no registry feature models same-path rewrites and the
        values are plausible numbers, so content/diversity/latency features
        stay at chance; the honest catch is volume (2 writes per 3 bits)."""
        clean = clean_traces(24)
        stego = correction_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name in (
            "calibrated:invisible_char_density",
            "calibrated:homoglyph_density",
            "calibrated:non_ascii_density",
            "calibrated:key_order_anomaly_fraction",
            "calibrated:number_format_anomaly_fraction",
            "calibrated:mean_latency",
            "calibrated:latency_std",
            "calibrated:unique_tools",
        ):
            res = evaluate_detector(by_name[name], stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"
        res = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert res["youden_j"] >= 0.8

    def test_observations_distinct(self) -> None:
        runs = correction_traces(3)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
