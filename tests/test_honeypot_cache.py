"""Tests for the composed honeypot-cache channel."""

from src.channels.corpus import clean_traces, honeypot_cache_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.honeypot import DECOY_TEXT
from src.channels.honeypot_cache import HoneypotCacheChannel, HoneypotCacheExecutor

MESSAGE = "HELLO"


class TestHoneypotCacheChannel:
    def test_protocol_receiver_recovers_real_payload(self) -> None:
        channel = HoneypotCacheChannel()
        executor = HoneypotCacheExecutor(channel=channel)
        traces = executor.run(channel.encode(MESSAGE))
        assert channel.decode(traces) == MESSAGE

    def test_content_inspector_finds_only_decoy(self) -> None:
        channel = HoneypotCacheChannel()
        executor = HoneypotCacheExecutor(channel=channel)
        traces = executor.run(channel.encode(MESSAGE))
        assert channel.decode_content(traces) == DECOY_TEXT.decode()

    def test_real_payload_never_in_any_content(self) -> None:
        """The composition's core property: the real message exists nowhere
        in files, arguments, or responses — only as cache state."""
        channel = HoneypotCacheChannel()
        executor = HoneypotCacheExecutor(channel=channel)
        traces = executor.run(channel.encode(MESSAGE))
        for t in traces:
            for value in t.arguments.values():
                assert MESSAGE not in str(value)

    def test_multiple_trials(self) -> None:
        for trial in range(5):
            channel = HoneypotCacheChannel(seed=500 + trial)
            executor = HoneypotCacheExecutor(channel=channel, seed=500 + trial)
            traces = executor.run(channel.encode(MESSAGE))
            assert channel.decode(traces) == MESSAGE
            assert channel.decode_content(traces) == DECOY_TEXT.decode()


class TestDetection:
    def test_honest_negatives_both_axes_fire(self) -> None:
        """Honest: the family is caught on BOTH content (zero-width decoy)
        and latency (cache alternation) — deception about what is
        recovered, not invisibility."""
        clean = clean_traces(24)
        stego = honeypot_cache_traces(18)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        content = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        latency = evaluate_detector(by_name["calibrated:latency_std"], stego, clean)
        assert content["youden_j"] == 1.0
        assert latency["youden_j"] >= 0.8

    def test_observations_distinct(self) -> None:
        runs = honeypot_cache_traces(3)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
