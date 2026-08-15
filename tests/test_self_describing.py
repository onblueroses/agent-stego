"""Tests for self-describing tool traces (in-trace channel headers)."""

import pytest

from src.channels.corpus import _cover_config
from src.channels.json_order import JsonConfigChannel
from src.channels.permutation import PermutationChannel
from src.channels.self_describing import (
    SelfDescribingChannel,
    TraceHeader,
    build_header,
    decode_trace_exhaustive,
    detect_trace_channel,
    parse_header,
    registered_channel,
    registered_keys,
)
from src.channels.table import TableChannel
from src.channels.text_composed import (
    AuditExecutor,
    FileContentChannel,
    HeaderExecutor,
    WriteReportExecutor,
)
from src.token_stego.text_channels import ZeroWidthChannel


class TestHeaderFormat:
    def test_build_header_text(self) -> None:
        assert (
            build_header("table/path", 5)
            == 'check_status({ "target": "stx1|table/path|5" })'
        )

    def test_round_trip(self) -> None:
        from src.channels.text_composed import HonestExecutor

        class HeaderOnly(HonestExecutor):
            def _extract_calls(self, prompt):  # type: ignore[override]
                return [("check_status", {"target": "stx1|table/path|5"})]

        assert parse_header(HeaderOnly().run("x").traces) == TraceHeader(
            channel_key="table/path", num_chars=5
        )

    def test_missing_header_is_none(self) -> None:
        assert parse_header([]) is None

    def test_bad_magic_is_none(self) -> None:
        from src.channels.text_composed import HonestExecutor

        class Fake(HonestExecutor):
            def _extract_calls(self, prompt):  # type: ignore[override]
                return []

        traces = Fake().run("no header here").traces
        assert parse_header(traces) is None

    def test_malformed_num_chars_is_none(self) -> None:
        from src.channels.text_composed import HonestExecutor

        class Fake(HonestExecutor):
            def _extract_calls(self, prompt):  # type: ignore[override]
                return [("check_status", {"target": "stx1|table/path|notanumber"})]

        assert parse_header(Fake().run("x").traces) is None


class TestWrapper:
    def test_table_round_trip_with_header(self) -> None:
        wrapped = SelfDescribingChannel("table/path", TableChannel(mode="path"))
        result = HeaderExecutor(AuditExecutor()).run(wrapped.encode("HI"))
        assert [t.tool_name for t in result.traces] == [
            "check_status",
            "read_file",
            "read_file",
        ]
        assert wrapped.decode(result.traces) == "HI"

    def test_permutation_round_trip(self) -> None:
        wrapped = SelfDescribingChannel("permutation", PermutationChannel())
        result = HeaderExecutor(AuditExecutor()).run(wrapped.encode("HI"))
        assert wrapped.decode(result.traces) == "HI"

    def test_file_content_round_trip(self) -> None:
        wrapped = SelfDescribingChannel(
            "file_content/zero_width",
            FileContentChannel(text_channel=ZeroWidthChannel()),
        )
        result = HeaderExecutor(WriteReportExecutor()).run(wrapped.encode("HELLO"))
        assert wrapped.decode(result.traces) == "HELLO"

    def test_json_config_round_trip(self) -> None:
        wrapped = SelfDescribingChannel(
            "config/json_key_order", JsonConfigChannel(cover=_cover_config(0))
        )
        result = HeaderExecutor(WriteReportExecutor()).run(wrapped.encode("HELLO"))
        assert wrapped.decode(result.traces) == "HELLO"

    def test_headerless_trace_decodes_empty(self) -> None:
        wrapped = SelfDescribingChannel("table/path", TableChannel(mode="path"))
        result = AuditExecutor().run(TableChannel(mode="path").encode("HI"))
        assert wrapped.decode(result.traces) == ""

    def test_wrong_key_rejected(self) -> None:
        wrapped = SelfDescribingChannel("table/path", TableChannel(mode="path"))
        result = HeaderExecutor(AuditExecutor()).run(wrapped.encode("HI"))
        # Rewrite the header to claim a different channel.
        for trace in result.traces:
            if trace.tool_name == "check_status":
                trace.arguments["target"] = "stx1|permutation|2"
        assert wrapped.decode(result.traces) == ""

    def test_length_validation_fails_closed(self) -> None:
        wrapped = SelfDescribingChannel("table/path", TableChannel(mode="path"))
        result = HeaderExecutor(AuditExecutor()).run(wrapped.encode("HI"))
        for trace in result.traces:
            if trace.tool_name == "check_status":
                trace.arguments["target"] = "stx1|table/path|99"
        assert wrapped.decode(result.traces) == ""


class TestDispatch:
    def test_detect_trace_channel(self) -> None:
        wrapped = SelfDescribingChannel("table/path", TableChannel(mode="path"))
        result = HeaderExecutor(AuditExecutor()).run(wrapped.encode("HI"))
        assert detect_trace_channel(result.traces) == ("table/path", "HI")

    def test_detect_none_without_header(self) -> None:
        result = AuditExecutor().run(TableChannel(mode="path").encode("HI"))
        assert detect_trace_channel(result.traces) is None

    def test_exhaustive_decode_includes_true_channel(self) -> None:
        wrapped = SelfDescribingChannel("table/path", TableChannel(mode="path"))
        result = HeaderExecutor(AuditExecutor()).run(wrapped.encode("HI"))
        candidates = decode_trace_exhaustive(result.traces)
        assert ("table/path", "HI") in candidates

    def test_registered_channels_instantiate(self) -> None:
        for key in registered_keys():
            channel = registered_channel(key)
            assert channel is not None

    def test_unregistered_key_rejected(self) -> None:
        with pytest.raises(KeyError):
            registered_channel("no/such/channel")
        with pytest.raises(KeyError):
            SelfDescribingChannel("no/such/channel", TableChannel(mode="path"))
