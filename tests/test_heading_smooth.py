"""Tests for the smooth 2-level heading channel (the heading-side gap)."""

from src.channels.corpus import (
    _HEADING_SMOOTH_COVERS,
    clean_traces,
    file_content_traces,
)
from src.channels.detector import build_detector_registry, evaluate_detector
from src.token_stego.text_channels import HeadingChannel

MESSAGE = b"HELLO"


class TestSmoothHeading:
    def test_round_trip(self) -> None:
        ch = HeadingChannel(levels=2)
        for cover in _HEADING_SMOOTH_COVERS:
            text = ch.encode(MESSAGE, cover)
            assert ch.decode(text) == MESSAGE

    def test_only_levels_one_and_two(self) -> None:
        ch = HeadingChannel(levels=2)
        text = ch.encode(MESSAGE, _HEADING_SMOOTH_COVERS[0])
        import re

        levels = {
            len(m.group(1)) for m in re.finditer(r"^(#{1,2})\s+", text, flags=re.M)
        }
        assert levels <= {1, 2}

    def test_capacity(self) -> None:
        ch = HeadingChannel(levels=2)
        assert ch.capacity(_HEADING_SMOOTH_COVERS[0]) >= 88  # 1 bit per heading


class TestDetection:
    def test_invisible_to_the_full_registry(self) -> None:
        """The heading-side precision gap: smooth #/## alternation is normal
        nesting, so the closure's skip rule never fires and all 18 features
        stay at chance."""
        clean = clean_traces(24)
        stego = file_content_traces("heading_smooth", 18, decoys=True)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name, det in by_name.items():
            res = evaluate_detector(det, stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"

    def test_observations_distinct(self) -> None:
        runs = file_content_traces("heading_smooth", 3, decoys=True)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
