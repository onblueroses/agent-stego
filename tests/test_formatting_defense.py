"""Tests for the formatting-consistency feature (closes the blind spot)."""

from src.channels.corpus import (
    _DATED_COVERS,
    clean_traces,
    file_content_traces,
)
from src.channels.datefmt import DateFormatChannel
from src.channels.detector import (
    _formatting_anomaly_fraction,
    build_detector_registry,
    evaluate_detector,
)
from src.token_stego.text_channels import CaseChannel, WhitespaceChannel

COVER = (
    "The report shows the audit completed. All checks passed and the team "
    "moved to the next phase. The status is healthy and the deployment is "
    "ready for the final review."
) * 4


class TestFeature:
    def test_clean_scores_zero(self) -> None:
        assert _formatting_anomaly_fraction([COVER]) == 0.0
        assert _formatting_anomaly_fraction(["2026/08/14: fixed."]) == 0.0

    def test_each_invisible_family_scores_positive(self) -> None:
        case = CaseChannel().encode(b"HELLO", COVER)
        date = DateFormatChannel().encode(b"HELLO", _DATED_COVERS[0])
        space = WhitespaceChannel().encode(b"HELLO", COVER)
        assert _formatting_anomaly_fraction([case]) > 0.0
        assert _formatting_anomaly_fraction([date]) > 0.0
        assert _formatting_anomaly_fraction([space]) > 0.0


class TestDetection:
    def test_closes_the_blind_spot(self) -> None:
        """The three previously-invisible families are caught at J=1.0 by
        the new feature."""
        clean = clean_traces(24)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        assert "calibrated:formatting_anomaly_fraction" in by_name
        for carrier in ("case", "datefmt", "whitespace", "heading"):
            stego = file_content_traces(carrier, 18, decoys=True)
            res = evaluate_detector(
                by_name["calibrated:formatting_anomaly_fraction"], stego, clean
            )
            assert res["youden_j"] == 1.0, f"{carrier}: {res['youden_j']}"

    def test_no_false_positives_on_content_families(self) -> None:
        """Character carriers keep single spaces/lowercase/no dates, so the
        feature stays at chance for them."""
        clean = clean_traces(24)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        for carrier in ("zero_width", "homoglyph", "nbsp"):
            stego = file_content_traces(carrier, 18, decoys=True)
            res = evaluate_detector(
                by_name["calibrated:formatting_anomaly_fraction"], stego, clean
            )
            assert res["auc"] < 0.8, f"{carrier}: {res['auc']}"
