"""Tests for the JSON key-order document carrier and its detection."""

import pytest

from src.channels.corpus import (
    _cover_config,
    clean_config_traces,
    clean_traces,
    json_key_order_traces,
)
from src.channels.detector import (
    build_detector_registry,
    evaluate_detector,
    extract_features,
)
from src.channels.json_order import JsonConfigChannel, JsonKeyOrderChannel, canonicalize
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import InsufficientCoverError

PAYLOAD = b"HELLO"


class TestJsonKeyOrderChannel:
    def test_round_trip(self) -> None:
        channel = JsonKeyOrderChannel()
        stego = channel.encode(PAYLOAD, _cover_config(0))
        assert channel.decode(stego) == PAYLOAD

    def test_capacity_sufficient(self) -> None:
        channel = JsonKeyOrderChannel()
        assert channel.capacity(_cover_config(0)) >= len(PAYLOAD) * 8 + 48

    def test_canonicalization_destroys_channel(self) -> None:
        """The blind monitor's view: sorted keys erase the payload."""
        channel = JsonKeyOrderChannel()
        stego = channel.encode(PAYLOAD, _cover_config(0))
        assert channel.decode(canonicalize(stego)) is None

    def test_clean_canonical_document_decodes_none(self) -> None:
        channel = JsonKeyOrderChannel()
        assert channel.decode(_cover_config(0)) is None

    def test_invalid_json_cover_rejected(self) -> None:
        channel = JsonKeyOrderChannel()
        with pytest.raises(ValueError):
            channel.encode_bits("1010", "not json at all")

    def test_insufficient_capacity_raises(self) -> None:
        channel = JsonKeyOrderChannel()
        small = '{"a": 1, "b": 2}'
        with pytest.raises(InsufficientCoverError):
            channel.encode(PAYLOAD, small)

    def test_values_unchanged_by_encoding(self) -> None:
        """Only key order changes; every value survives byte-identical values."""
        channel = JsonKeyOrderChannel()
        import json

        stego = channel.encode(PAYLOAD, _cover_config(1))
        original = json.loads(_cover_config(1))
        encoded = json.loads(stego)
        assert sorted(encoded) == sorted(original)
        assert encoded["version"] == original["version"]
        assert encoded["logging"]["level"] == "info"

    def test_partial_field_alignment(self) -> None:
        """Placed bits must survive truncation even when the stream does not
        align to per-object field boundaries (regression: the final partial
        field used to left-pad, shifting the stream)."""
        channel = JsonKeyOrderChannel()
        bits = "1" * 48  # cuts the config's 4th field at 6 bits
        stego = channel.encode_bits(bits, _cover_config(0))
        assert channel.decode_bits(stego)[:48] == bits


class TestJsonConfigChannel:
    def test_round_trip_via_executor_with_decoys(self) -> None:
        channel = JsonConfigChannel(
            cover=_cover_config(0), decoy_paths=("config.json", "notes.md")
        )
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        assert [t.tool_name for t in result.traces] == [
            "read_file",
            "read_file",
            "write_file",
        ]
        assert channel.decode(result.traces) == "HELLO"

    def test_canonicalizing_monitor_is_blind(self) -> None:
        channel = JsonConfigChannel(cover=_cover_config(0))
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        content = result.traces[-1].arguments["content"]
        assert channel.channel.decode(canonicalize(content)) is None


class TestCorpusIntegration:
    def test_clean_configs_are_canonical(self) -> None:
        for traces in clean_config_traces(4):
            features = extract_features(traces)
            assert features["key_order_anomaly_fraction"] == 0.0

    def test_stego_configs_have_key_order_signal(self) -> None:
        for traces in json_key_order_traces(4):
            features = extract_features(traces)
            assert features["key_order_anomaly_fraction"] > 0.0

    def test_distinct_observations(self) -> None:
        stego = json_key_order_traces(6)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 3))
                for t in traces
            )
            for traces in stego
        ]
        assert len(set(keys)) == len(keys)

    def test_key_order_detector_catches_family_isolated_from_diversity(self) -> None:
        """Decoyed stego matches the clean call shape, so only the structural
        detector separates the families."""
        clean = clean_traces(24)
        stego = json_key_order_traces(12)  # decoys on by default
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:key_order_anomaly_fraction"], stego, clean
        )
        assert stats["youden_j"] == 1.0
        diversity = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert diversity["youden_j"] == 0.0


def json_sorted(obj: object) -> str:
    import json

    return json.dumps(obj, sort_keys=True)
