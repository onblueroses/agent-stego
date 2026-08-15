"""Tests for the case channel (bits in letter case)."""

from src.channels.corpus import clean_traces, file_content_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.token_stego.text_channels import (
    CaseChannel,
    _eligible_case_indices,
    exhaustive_decode,
)

COVER = (
    "The report shows the audit completed. All checks passed and the team "
    "moved to the next phase. The status is healthy and the deployment is "
    "ready for the final review."
)
MESSAGE = b"HELLO"


class TestCaseChannel:
    def test_round_trip(self) -> None:
        ch = CaseChannel()
        text = ch.encode(MESSAGE, COVER)
        assert ch.decode(text) == MESSAGE

    def test_sentence_initial_capitals_preserved(self) -> None:
        """Natural sentence capitals are never read as payload."""
        ch = CaseChannel()
        text = ch.encode(MESSAGE, COVER)
        # sentence-initial letters stay uppercase from the cover
        assert text[0] == "T"
        assert text[COVER.index("All")] == "A"

    def test_capacity_counts_mid_sentence_letters(self) -> None:
        eligible = len(_eligible_case_indices(COVER))
        assert ch_capacity(COVER) == eligible
        assert eligible >= 88  # enough for HELLO's framed payload

    def test_exhaustive_monitor_finds_it(self) -> None:
        """The case channel is registered, so an exhaustive monitor that
        scans all carriers recovers the payload (its honest catch)."""
        ch = CaseChannel()
        text = ch.encode(MESSAGE, COVER)
        assert exhaustive_decode(text) == MESSAGE

    def test_lowercase_cover_round_trip(self) -> None:
        ch = CaseChannel()
        text = ch.encode(MESSAGE, COVER.lower())
        assert ch.decode(text) == MESSAGE


def ch_capacity(cover: str) -> int:
    return CaseChannel().capacity(cover)


class TestDetection:
    def test_registry_has_no_case_feature(self) -> None:
        """The headline honest finding: an all-ASCII case-flipped report is
        invisible to every ORIGINAL registered feature (density, structure,
        lexical, timing). The formatting-consistency feature added in round
        66 closes this blind spot — it is the measured defense, so it is
        excluded here and asserted to catch the family in
        test_formatting_defense.py."""
        clean = clean_traces(24)
        stego = file_content_traces("case", 18, decoys=True)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for name, det in by_name.items():
            if name == "calibrated:formatting_anomaly_fraction":
                continue
            res = evaluate_detector(det, stego, clean)
            assert res["auc"] < 0.8, f"{name} not blind: {res['auc']}"

    def test_observations_distinct(self) -> None:
        runs = file_content_traces("case", 3, decoys=True)
        keys = [
            tuple(
                (t.tool_name, tuple(sorted(t.arguments)), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)
