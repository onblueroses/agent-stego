"""Tests for the heading-level channel."""

from src.channels.corpus import _HEADING_COVERS, clean_traces, file_content_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.token_stego.text_channels import HeadingChannel

MESSAGE = b"HELLO"


class TestHeadingChannel:
    def test_round_trip(self) -> None:
        ch = HeadingChannel()
        for cover in _HEADING_COVERS:
            text = ch.encode(MESSAGE, cover)
            assert ch.decode(text) == MESSAGE

    def test_all_four_levels_used(self) -> None:
        ch = HeadingChannel()
        bits = "".join(f"{i:02b}" for i in range(4)) * 12
        text = ch.encode_bits(bits, _HEADING_COVERS[0])
        levels = {
            len(m.group(1))
            for m in __import__("re").finditer(
                r"^(#{1,4})\s+", text, flags=__import__("re").MULTILINE
            )
        }
        assert levels == {1, 2, 3, 4}

    def test_capacity_counts_headings(self) -> None:
        n = HeadingChannel().capacity(_HEADING_COVERS[0])
        assert n >= 44  # enough for HELLO's 88 framed bits


class TestDetection:
    def test_invisible_to_original_registry(self) -> None:
        """Heading levels are unmodeled by every ORIGINAL feature; the
        round-68 extension of formatting_anomaly_fraction (heading-depth
        consistency) closes this axis and is asserted in
        test_formatting_defense.py."""
        clean = clean_traces(24)
        stego = file_content_traces("heading", 18, decoys=True)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name, det in by_name.items():
            if name == "calibrated:formatting_anomaly_fraction":
                continue
            res = evaluate_detector(det, stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"

    def test_observations_distinct(self) -> None:
        runs = file_content_traces("heading", 3, decoys=True)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
