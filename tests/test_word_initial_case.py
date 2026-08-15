"""Tests for the word-initial-case channel (the precision gap)."""

from src.channels.corpus import clean_traces, file_content_traces
from src.channels.detector import (
    _formatting_anomaly_fraction,
    build_detector_registry,
    evaluate_detector,
)
from src.token_stego.text_channels import WordInitialChannel, _word_initial_indices

COVER = (
    "The report shows the audit completed. All checks passed and the team "
    "moved to the next phase. The status is healthy and the deployment is "
    "ready for the final review."
) * 4
MESSAGE = b"HELLO"


class TestWordInitialChannel:
    def test_round_trip(self) -> None:
        ch = WordInitialChannel()
        text = ch.encode(MESSAGE, COVER)
        assert ch.decode(text) == MESSAGE

    def test_capacity_counts_eligible_word_initials(self) -> None:
        n = len(_word_initial_indices(COVER))
        assert WordInitialChannel().capacity(COVER) == n
        assert n >= 88  # enough for HELLO's framed payload

    def test_sentence_initial_capitals_never_bits(self) -> None:
        ch = WordInitialChannel()
        text = ch.encode(MESSAGE, COVER)
        # The cover's sentence-initial capitals survive untouched.
        assert text[0] == "T"
        assert text[COVER.index("All")] == "A"


class TestPrecisionGap:
    def test_invisible_to_the_full_registry_including_the_closure(self) -> None:
        """The closure's case sub-score counts only MID-WORD flips (by
        design, to avoid proper-noun false positives); word-initial flips
        are invisible to it and to every other feature."""
        clean = clean_traces(24)
        stego = file_content_traces("wordcase", 18, decoys=True)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name, det in by_name.items():
            res = evaluate_detector(det, stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"

    def test_gap_is_structurally_unclosable(self) -> None:
        """A flipped word-initial 'Report' is byte-identical to a proper
        noun 'Report': the closure scores BOTH zero, so no syntactic rule
        can separate the channel from natural proper-noun text."""
        flip_cover = "the report is final and the audit passed all checks today"
        flipped = WordInitialChannel().encode_bits("1" * 10, flip_cover)
        proper_noun = "The Report is Final for review"
        assert _formatting_anomaly_fraction([flipped]) == 0.0
        assert _formatting_anomaly_fraction([proper_noun]) == 0.0

    def test_observations_distinct(self) -> None:
        runs = file_content_traces("wordcase", 3, decoys=True)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
