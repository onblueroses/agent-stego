"""Tests for the honeypot (decoy-under-real) channel."""

import json

from src.channels.corpus import clean_traces, honeypot_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.honeypot import DECOY_TEXT, HoneypotChannel
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import all_channels, exhaustive_decode

MESSAGE = "HELLO"


def _artifact() -> str:
    return HoneypotChannel().artifact(MESSAGE)


class TestHoneypotChannel:
    def test_receiver_recovers_real_payload(self) -> None:
        channel2 = HoneypotChannel()
        result = ReportWithDecoysExecutor().run(channel2.encode(MESSAGE))
        assert channel2.decode(result.traces) == MESSAGE

    def test_naive_monitor_recovers_decoy_not_real(self) -> None:
        artifact = _artifact()
        found = exhaustive_decode(artifact)
        assert found == DECOY_TEXT
        assert found != MESSAGE.encode("ascii")

    def test_thorough_monitor_finds_both(self) -> None:
        artifact = _artifact()
        found = [
            payload
            for text_channel in all_channels()
            if (payload := text_channel.decode(artifact)) is not None
        ]
        assert DECOY_TEXT in found
        assert MESSAGE.encode("ascii") in found

    def test_homoglyph_decode_isolated(self) -> None:
        artifact = _artifact()
        channel = HoneypotChannel()
        assert channel.decode_homoglyph(artifact) == MESSAGE
        assert channel.decode_zero_width(artifact) == DECOY_TEXT.decode()


class TestDetection:
    def test_both_scanners_fire(self) -> None:
        """Honest: the honeypot raises detectability — the decoy's zero-width
        signal and the real homoglyph signal both separate the family."""
        clean = clean_traces(24)
        stego = honeypot_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        invisible = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        homoglyph = evaluate_detector(
            by_name["calibrated:homoglyph_density"], stego, clean
        )
        assert invisible["youden_j"] == 1.0
        assert homoglyph["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = honeypot_traces(3)
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
