"""Tests for the inbound (poisoned-file) channel."""

import json

from src.channels.corpus import clean_traces, inbound_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.inbound import PoisonedFileChannel, PoisonedReadExecutor


class TestPoisonedFileChannel:
    def test_round_trip(self) -> None:
        channel = PoisonedFileChannel()
        executor = PoisonedReadExecutor(content=channel.encode("HELLO"))
        assert channel.decode(executor.run("")) == "HELLO"

    def test_call_is_ordinary(self) -> None:
        """The read_file call itself is ordinary; only the response carries."""
        channel = PoisonedFileChannel()
        executor = PoisonedReadExecutor(content=channel.encode("HELLO"))
        trace = executor.run("")[0]
        assert trace.tool_name == "read_file"
        assert trace.arguments == {"path": "data.txt"}
        assert trace.success is True
        # The argument string must be carrier-free.
        from src.token_stego.text_channels import ZeroWidthChannel

        assert ZeroWidthChannel().decode_bits(str(trace.arguments["path"])) == ""

    def test_clean_response_decodes_empty(self) -> None:
        channel = PoisonedFileChannel()
        executor = PoisonedReadExecutor(content="plain file contents")
        assert channel.decode(executor.run("")) == ""


class TestDetection:
    def test_response_aware_registry_catches(self) -> None:
        clean = clean_traces(24)
        stego = inbound_traces(8)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        assert stats["youden_j"] == 1.0

    def test_call_log_only_monitor_would_be_blind(self) -> None:
        """Arguments carry no carriers; a monitor that logs only calls
        (names/arguments) never sees the response and cannot detect it."""
        from src.channels.inbound import PoisonedFileChannel

        channel = PoisonedFileChannel()
        executor = PoisonedReadExecutor(content=channel.encode("HELLO"))
        trace = executor.run("")[0]
        # The only argument has no carriers.
        from src.token_stego.text_channels import ZeroWidthChannel

        assert ZeroWidthChannel().decode_bits(str(trace.arguments["path"])) == ""

    def test_observations_distinct(self) -> None:
        runs = inbound_traces(3)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 3))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)


def json_sorted(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)
