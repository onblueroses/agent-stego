"""Tests for multi-stage cross-surface composition."""

from src.channels.corpus import clean_traces, multistage_traces
from src.channels.detector import (
    build_detector_registry,
    evaluate_detector,
    evaluate_registry,
)
from src.channels.multistage import (
    MultiStageChannel,
    MultiStageExecutor,
    _split_bits,
)

MESSAGE = "HELLO"


def _run(message: str = MESSAGE) -> tuple[list, MultiStageChannel]:
    channel = MultiStageChannel()
    executor = MultiStageExecutor(start_time=9_500_000.0)
    return executor.run(channel.encode(message)), channel


class TestSplit:
    def test_split_exhaustive(self) -> None:
        assert sum(_split_bits(88, 3)) == 88
        assert len(_split_bits(88, 3)) == 3


class TestRoundTrip:
    def test_joint_decode_exact(self) -> None:
        traces, channel = _run()
        assert channel.decode(traces) == MESSAGE

    def test_stage_bit_counts(self) -> None:
        traces, channel = _run()
        surface = channel.surface_bits(traces)
        counts = channel.stage_bit_counts(len(MESSAGE) * 8)
        for stage, expected in counts.items():
            assert len(surface[stage]) >= expected, stage

    def test_all_surfaces_contribute(self) -> None:
        traces, channel = _run()
        surface = channel.surface_bits(traces)
        for stage in channel.stages:
            assert len(surface[stage]) > 0, stage


class TestPerSurfaceBlindness:
    def test_no_single_surface_is_self_framing(self) -> None:
        """The frame spans all surfaces; each surface's fragment alone has no
        magic, so per-surface monitors are blind by construction."""
        traces, channel = _run()
        for stage in channel.stages:
            assert channel.decode_surface(traces, stage) == "", stage

    def test_reordered_stages_fail(self) -> None:
        """The stage map is the coordination key; wrong order fails closed."""
        traces, channel = _run()
        bits = channel.surface_bits(traces)
        wrong_order = "".join(
            bits[stage] for stage in ("file_content", "key_order", "tool_choice")
        )
        from src.token_stego.text_channels import _unframe_bits

        assert _unframe_bits(wrong_order) is None


class TestCorpusAndDetection:
    def test_corpus_family_caught_by_registry(self) -> None:
        """Honest: each stage leaves its surface's signature; the full
        registry separates the family."""
        clean = clean_traces(24)
        stego = multistage_traces(8)
        detectors = build_detector_registry(clean)
        matrix = evaluate_registry(detectors, stego, clean)
        best_j = max(stats["youden_j"] for stats in matrix.values())
        assert best_j == 1.0

    def test_call_count_catches_tool_stage(self) -> None:
        """The tool-choice stage inflates call count far past clean runs."""
        clean = clean_traces(24)
        stego = multistage_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = multistage_traces(3)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)


def json_sorted(obj: object) -> str:
    import json

    return json.dumps(obj, sort_keys=True)
