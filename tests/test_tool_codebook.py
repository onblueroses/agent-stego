"""Tests for the tool-codebook channel."""

from src.channels.corpus import clean_traces, tool_codebook_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.tool_codebook import (
    TOOL_MENU,
    ToolCodebookChannel,
    ToolCodebookExecutor,
)

MESSAGE = "HELLO"


class TestToolCodebook:
    def test_encode_decode_round_trip(self) -> None:
        channel = ToolCodebookChannel()
        executor = ToolCodebookExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        assert channel.decode(traces) == MESSAGE

    def test_every_call_is_a_real_tool(self) -> None:
        channel = ToolCodebookChannel()
        executor = ToolCodebookExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        menu = {n for n, _ in TOOL_MENU}
        assert {t.tool_name for t in traces} <= menu

    def test_round_trips_many_seeds(self) -> None:
        for trial in range(5):
            channel = ToolCodebookChannel(seed=24_000 + trial)
            executor = ToolCodebookExecutor(seed=24_000 + trial)
            traces = executor.run(channel.encode(MESSAGE))
            assert channel.decode(traces) == MESSAGE


class TestDetection:
    def test_content_and_diversity_blind(self) -> None:
        """The calls are ordinary and four tools is not suspicious
        (unique_tools flags FEWER tools), so content/diversity/latency
        features stay at chance; the honest tell is volume."""
        clean = clean_traces(24)
        stego = tool_codebook_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name in (
            "calibrated:invisible_char_density",
            "calibrated:homoglyph_density",
            "calibrated:non_ascii_density",
            "calibrated:key_order_anomaly_fraction",
            "calibrated:unique_tools",
            "calibrated:single_tool_fraction",
            "calibrated:mean_latency",
            "calibrated:latency_std",
        ):
            res = evaluate_detector(by_name[name], stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"
        res = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert res["youden_j"] >= 0.8

    def test_observations_distinct(self) -> None:
        runs = tool_codebook_traces(3)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
