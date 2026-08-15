"""Tests for the tool-trace detector suite."""

from src.channels.corpus import (
    argument_carrier_traces,
    clean_audit_traces,
    clean_report_traces,
    clean_traces,
    file_content_traces,
    table_channel_traces,
)
from src.channels.detector import (
    build_detector_registry,
    calibrate_thresholds,
    evaluate_detector,
    evaluate_registry,
    extract_features,
)


class TestFeatureExtraction:
    def test_clean_report_has_zero_content_signal(self) -> None:
        for traces in clean_report_traces(4):
            features = extract_features(traces)
            assert features["invisible_char_density"] == 0.0
            assert features["nbsp_density"] == 0.0
            assert features["homoglyph_density"] == 0.0
            assert features["non_ascii_density"] == 0.0

    def test_surface_features_present(self) -> None:
        traces = clean_audit_traces(1)[0]
        features = extract_features(traces)
        assert features["call_count"] == len(traces)
        assert features["unique_tools"] == 2  # list_files + read_file

    def test_zero_width_content_is_invisible_signal(self) -> None:
        traces = file_content_traces("zero_width", 1)[0]
        assert extract_features(traces)["invisible_char_density"] > 0.0

    def test_nbsp_content_is_nbsp_signal(self) -> None:
        traces = file_content_traces("nbsp", 1)[0]
        assert extract_features(traces)["nbsp_density"] > 0.0

    def test_nfd_content_is_combining_signal(self) -> None:
        traces = file_content_traces("nfd", 1)[0]
        assert extract_features(traces)["combining_mark_density"] > 0.0

    def test_homoglyph_argument_is_homoglyph_signal(self) -> None:
        traces = argument_carrier_traces(1)[0]
        assert extract_features(traces)["homoglyph_density"] > 0.0


class TestCalibration:
    def test_thresholds_calibrated_on_clean_only(self) -> None:
        clean = clean_traces(12)
        thresholds = calibrate_thresholds(clean, ["nbsp_density"])
        # Clean traces have no NBSP; the calibrated threshold must flag none.
        assert all(
            extract_features(t)["nbsp_density"] <= thresholds["nbsp_density"]
            for t in clean
        )

    def test_registry_zero_false_positive_on_clean(self) -> None:
        clean = clean_traces(12)
        detectors = build_detector_registry(clean)
        for detector in detectors:
            assert all(detector.score(t) == 0.0 for t in clean), detector.name


class TestDetection:
    def test_invisible_detector_catches_zero_width_file(self) -> None:
        clean = clean_traces(12)
        stego = file_content_traces("zero_width", 8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        assert stats["youden_j"] == 1.0

    def test_nbsp_detector_catches_nbsp_file(self) -> None:
        clean = clean_traces(12)
        stego = file_content_traces("nbsp", 8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:nbsp_density"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_combining_detector_catches_nfd_file(self) -> None:
        clean = clean_traces(12)
        stego = file_content_traces("nfd", 8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:combining_mark_density"], stego, clean
        )
        assert stats["youden_j"] == 1.0

    def test_homoglyph_detector_catches_argument_carrier(self) -> None:
        clean = clean_traces(12)
        stego = argument_carrier_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:homoglyph_density"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_tool_diversity_detector_catches_table_channel(self) -> None:
        clean = clean_traces(12)
        stego = table_channel_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_diversity_detector_catches_naive_file_content(self) -> None:
        """Naive stego writers emit one call; the diversity scan catches them."""
        clean = clean_traces(12)
        stego = file_content_traces("zero_width", 8)  # no decoys
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert stats["youden_j"] == 1.0

    def test_diversity_detector_blind_to_decoyed_file_content(self) -> None:
        """Decoy reads restore the natural call pattern; diversity is blind."""
        clean = clean_traces(12)
        stego = file_content_traces("zero_width", 8, decoys=True)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert stats["youden_j"] == 0.0

    def test_call_count_is_blind_under_5sigma(self) -> None:
        """Surface call-count does not catch the file-content family at the
        5-sigma calibration: clean writers emit 3 calls, stego writers 1, and
        the calibrated threshold sits far above both."""
        clean = clean_traces(12)
        stego = file_content_traces("zero_width", 8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:call_count"], stego, clean)
        assert stats["youden_j"] == 0.0

    def test_latency_detector_is_blind_to_content(self) -> None:
        """Surface timing carries no content signal in this corpus (J = 0)."""
        clean = clean_traces(12)
        stego = file_content_traces("zero_width", 8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:mean_latency"], stego, clean)
        assert stats["youden_j"] == 0.0

    def test_registry_matrix_shape(self) -> None:
        clean = clean_traces(12)
        stego = file_content_traces("zero_width", 8)
        detectors = build_detector_registry(clean)
        matrix = evaluate_registry(detectors, stego, clean)
        assert set(matrix) == {d.name for d in detectors}
        assert all(
            set(stats) == {"auc", "tpr", "fpr", "youden_j"} for stats in matrix.values()
        )


class TestCorpus:
    def test_observations_are_distinct(self) -> None:
        """No exact duplicate traces within a family (integrity rule)."""
        stego = file_content_traces("zero_width", 6)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 3))
                for t in traces
            )
            for traces in stego
        ]
        assert len(set(keys)) == len(keys)

    def test_clean_traces_mixed(self) -> None:
        clean = clean_traces(10)
        assert len(clean) == 10
        # Some writers (write_file present) and some auditors.
        assert any(any(t.tool_name == "write_file" for t in traces) for traces in clean)
        assert any(
            all(t.tool_name in {"list_files", "read_file"} for t in traces)
            and any(t.tool_name == "list_files" for t in traces)
            for traces in clean
        )


def json_sorted(obj: object) -> str:
    import json

    return json.dumps(obj, sort_keys=True)
