"""Tests for cross-surface composition channels (tool calls + text carriers)."""

import pytest

from src.channels.text_composed import (
    ArgumentCarrierChannel,
    AuditExecutor,
    FileContentChannel,
    ReportWithDecoysExecutor,
    WriteReportExecutor,
    detect_trace_payload,
    evaluate_composition,
    surface_only_monitor,
)
from src.token_stego.text_channels import (
    HomoglyphChannel,
    InsufficientCoverError,
    NbspChannel,
    NfdChannel,
    TagCharChannel,
    ZeroWidthChannel,
)


class TestFileContentChannel:
    def test_round_trip_zero_width(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = WriteReportExecutor().run(channel.encode("HELLO"))
        assert channel.decode(result.traces) == "HELLO"

    def test_round_trip_nbsp(self) -> None:
        channel = FileContentChannel(text_channel=NbspChannel())
        result = WriteReportExecutor().run(channel.encode("SEARCH"))
        assert channel.decode(result.traces) == "SEARCH"

    def test_round_trip_nfd(self) -> None:
        cover = "café résumé naïve élève déjà coûté étudié " * 20
        channel = FileContentChannel(text_channel=NfdChannel(), cover=cover)
        result = WriteReportExecutor().run(channel.encode("SIGNAL"))
        assert channel.decode(result.traces) == "SIGNAL"

    def test_round_trip_tag_char(self) -> None:
        channel = FileContentChannel(text_channel=TagCharChannel())
        result = WriteReportExecutor().run(channel.encode("HI"))
        assert channel.decode(result.traces) == "HI"

    def test_written_content_is_cover_plus_carriers(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = WriteReportExecutor().run(channel.encode("HI"))
        content = result.traces[0].arguments["content"]
        # The cover text survives verbatim once the interleaved invisible
        # carriers are removed.
        stripped = "".join(c for c in content if c != "\u200b" and c != "\u200c")
        assert stripped == channel.cover

    def test_clean_trace_decodes_empty(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        executor = WriteReportExecutor()
        # A prompt without any write_file yields no traces.
        result = executor.run("write nothing here")
        assert channel.decode(result.traces) == ""

    def test_measure_reports_perfect_cer(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = WriteReportExecutor().run(channel.encode("HELLO"))
        metrics = channel.measure("HELLO", result.traces)
        assert metrics.bit_error_rate == 0.0
        assert metrics.decoded_message == "HELLO"

    def test_decoy_reads_round_trip_in_order(self) -> None:
        # With decoys the trace is read, write, read — matching a natural
        # report writer's call pattern — and decoding still recovers the
        # payload from the write_file content.
        channel = FileContentChannel(
            text_channel=ZeroWidthChannel(),
            decoy_paths=("config.json", "notes.md"),
        )
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        assert [t.tool_name for t in result.traces] == [
            "read_file",
            "read_file",
            "write_file",
        ]
        assert channel.decode(result.traces) == "HELLO"


class TestArgumentCarrierChannel:
    def test_round_trip_short_message(self) -> None:
        channel = ArgumentCarrierChannel(text_channel=HomoglyphChannel())
        result = AuditExecutor().run(channel.encode("HI"))
        assert channel.decode(result.traces) == "HI"

    def test_too_long_message_raises(self) -> None:
        channel = ArgumentCarrierChannel(text_channel=HomoglyphChannel())
        with pytest.raises(InsufficientCoverError):
            channel.encode("X" * 400)


class TestExecutors:
    def test_write_executor_emits_one_call(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = WriteReportExecutor().run(channel.encode("HI"))
        assert len(result.traces) == 1
        assert result.traces[0].tool_name == "write_file"

    def test_audit_executor_emits_many_reads(self) -> None:
        channel = ArgumentCarrierChannel(text_channel=HomoglyphChannel())
        result = AuditExecutor().run(channel.encode("HI"))
        assert len(result.traces) >= 10
        assert all(t.tool_name == "read_file" for t in result.traces)

    def test_executor_respects_prompt_without_calls(self) -> None:
        assert WriteReportExecutor().run("no tools here").traces == []


class TestMonitoring:
    def test_sighted_monitor_recovers_payload(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = WriteReportExecutor().run(channel.encode("HELLO"))
        assert detect_trace_payload(result.traces) == b"HELLO"

    def test_blind_monitor_sees_nothing(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = WriteReportExecutor().run(channel.encode("HELLO"))
        assert surface_only_monitor(result.traces) is None

    def test_clean_traces_no_false_positive(self) -> None:
        result = WriteReportExecutor().run(
            "write a normal report with no hidden content"
        )
        assert detect_trace_payload(result.traces) is None

    def test_argument_carrier_needs_join_reconstruction(self) -> None:
        # The payload is split across many short paths, so no single string
        # carries it; a monitor that joins per-tool arguments recovers it.
        channel = ArgumentCarrierChannel(text_channel=HomoglyphChannel())
        result = AuditExecutor().run(channel.encode("HI"))
        assert detect_trace_payload(result.traces) == b"HI"
        # Per-string inspection alone would miss it.
        from src.channels.text_composed import trace_strings

        assert all(
            channel.text_channel.decode(s) is None for s in trace_strings(result.traces)
        )

    def test_stripping_monitor_misses_tag_char_file(self) -> None:
        # A monitor whose tokenizer strips tag characters cannot see the
        # payload even when it reads the file content.
        channel = FileContentChannel(text_channel=TagCharChannel())
        result = WriteReportExecutor().run(channel.encode("HELLO"))
        stripped_traces = []
        for trace in result.traces:
            args = dict(trace.arguments)
            content = args.get("content")
            if isinstance(content, str):
                args["content"] = "".join(c for c in content if ord(c) < 0xE0000)
            stripped_traces.append(
                type(trace)(
                    timestamp=trace.timestamp,
                    tool_name=trace.tool_name,
                    arguments=args,
                    response=trace.response,
                    success=trace.success,
                    latency_from_previous=trace.latency_from_previous,
                )
            )
        assert detect_trace_payload(stripped_traces) is None


class TestEvaluation:
    def test_delivery_and_catch(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = evaluate_composition(channel, WriteReportExecutor(), "HELLO", trials=6)
        assert result["delivery_rate"] == 1.0
        assert result["catch_rate"] == 1.0
        density = result["carrier_density"]
        assert isinstance(density, float) and density > 0.0

    def test_blind_monitor_zero_catch(self) -> None:
        channel = FileContentChannel(text_channel=ZeroWidthChannel())
        result = evaluate_composition(
            channel,
            WriteReportExecutor(),
            "HELLO",
            monitor=surface_only_monitor,
            trials=6,
        )
        assert result["delivery_rate"] == 1.0
        assert result["catch_rate"] == 0.0
