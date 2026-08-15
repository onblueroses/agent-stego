"""Tests for the multi-model routing channel."""

from src.channels.corpus import clean_traces, routing_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.routing import (
    BITS_PER_REQUEST,
    MODEL_MENU,
    RouteChannel,
    RouteExecutor,
)

MESSAGE = "HELLO"


class TestRouteChannel:
    def test_encode_decode_round_trip(self) -> None:
        channel = RouteChannel()
        executor = RouteExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        assert channel.decode(traces) == MESSAGE

    def test_every_route_is_a_real_model(self) -> None:
        channel = RouteChannel()
        executor = RouteExecutor()
        traces = executor.run(channel.encode(MESSAGE))
        ids = {t.model_id for t in traces if t.model_id is not None}
        menu_ids = {mid for mid, _, _ in MODEL_MENU}
        assert ids <= menu_ids

    def test_two_bits_per_request(self) -> None:
        assert BITS_PER_REQUEST == 2

    def test_round_trips_many_seeds(self) -> None:
        for trial in range(5):
            channel = RouteChannel(seed=12_000 + trial)
            executor = RouteExecutor(seed=12_000 + trial)
            traces = executor.run(channel.encode(MESSAGE))
            assert channel.decode(traces) == MESSAGE


class TestDetection:
    def test_content_scanners_blind(self) -> None:
        """The routed tool calls are ordinary; the route is metadata not in
        any argument, so content/structure features stay at chance."""
        clean = clean_traces(24)
        stego = routing_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name in (
            "calibrated:invisible_char_density",
            "calibrated:homoglyph_density",
            "calibrated:non_ascii_density",
            "calibrated:key_order_anomaly_fraction",
            "calibrated:unique_tools",
        ):
            res = evaluate_detector(by_name[name], stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"

    def test_volume_signature_fires(self) -> None:
        """Honest negative: one routed request per 2 bits, so call-count
        scanners catch the volume (the route is not in the content, but the
        number of calls is)."""
        clean = clean_traces(24)
        stego = routing_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        res = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert res["youden_j"] >= 0.8

    def test_observations_distinct(self) -> None:
        runs = routing_traces(3)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
