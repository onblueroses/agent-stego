"""Tests for the spread-spectrum multi-carrier channel."""

import json

from src.channels.corpus import (
    _LETTER_RICH_COVERS,
    _PLAIN_COVERS,
    _SPACE_RICH_COVERS,
    clean_traces,
    spread_traces,
)
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.spread import SpreadChannel, strip_carrier

MESSAGE = "HELLO"


def _docs(channel: SpreadChannel, i: int = 0) -> list[str]:
    covers = [_PLAIN_COVERS[0], _SPACE_RICH_COVERS[0], _LETTER_RICH_COVERS[0]]
    return channel.encode(MESSAGE, covers)


class TestSpreadChannel:
    def test_round_trip(self) -> None:
        channel = SpreadChannel()
        assert channel.decode(_docs(channel)) == b"HELLO"

    def test_strip_one_carrier_still_delivers(self) -> None:
        """A monitor canonicalizing exactly one carrier class cannot kill it."""
        channel = SpreadChannel()
        for stripped in channel.carriers:
            assert channel.decode_after_strip(_docs(channel), stripped) == b"HELLO", (
                stripped
            )

    def test_strip_all_kills(self) -> None:
        channel = SpreadChannel()
        docs = [
            strip_carrier(carrier, doc)
            for carrier, doc in zip(channel.carriers, _docs(channel))
        ]
        assert channel.decode(docs) is None

    def test_wrong_doc_count_fails(self) -> None:
        channel = SpreadChannel()
        assert channel.decode(_docs(channel)[:2]) is None


class TestDetection:
    def test_multiple_scanners_fire(self) -> None:
        """Honest: redundancy multiplies the signal — several content
        scanners separate the family at J = 1.0."""
        clean = clean_traces(24)
        stego = spread_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        invisible = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        nbsp = evaluate_detector(by_name["calibrated:nbsp_density"], stego, clean)
        homoglyph = evaluate_detector(
            by_name["calibrated:homoglyph_density"], stego, clean
        )
        assert invisible["youden_j"] == 1.0
        assert nbsp["youden_j"] == 1.0
        assert homoglyph["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = spread_traces(3)
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
