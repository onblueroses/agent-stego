"""Tests for multi-turn file-content exfiltration."""

from src.channels.corpus import clean_traces, multiturn_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.multiturn import GAP, MultiTurnFileExfil, run_multi_turn
from src.channels.text_composed import ReportWithDecoysExecutor

MESSAGE = "HELLOWORLD"


def _run(message: str = MESSAGE) -> tuple[list, MultiTurnFileExfil]:
    exfil = MultiTurnFileExfil()
    prompts = exfil.get_turn_prompts(message)
    executor = ReportWithDecoysExecutor(start_time=9_000_000.0)
    return run_multi_turn(prompts, executor), exfil


class TestTurnPrompts:
    def test_one_prompt_per_character(self) -> None:
        exfil = MultiTurnFileExfil()
        prompts = exfil.get_turn_prompts(MESSAGE)
        assert len(prompts) == len(MESSAGE)
        for index, prompt in enumerate(prompts):
            assert f"chunk_{index}.md" in prompt
            assert f"Turn {index + 1} of {len(MESSAGE)}" in prompt

    def test_non_alpha_filtered(self) -> None:
        """Non-alphabetic characters are stripped before encoding."""
        exfil = MultiTurnFileExfil()
        prompts = exfil.get_turn_prompts("A\x00 B-1")
        assert len(prompts) == 2  # only A and B survive


class TestRoundTrip:
    def test_full_recovery(self) -> None:
        traces, exfil = _run()
        assert exfil.decode(traces) == MESSAGE

    def test_encode_joined_contract(self) -> None:
        exfil = MultiTurnFileExfil()
        assert "HELLO" not in exfil.encode("HI")  # joined prompts, not plaintext
        assert "---" in exfil.encode("HI")


class TestResilience:
    def test_dropped_turn_leaves_exactly_one_gap(self) -> None:
        traces, exfil = _run()
        # Drop turn 3 (its write call and decoys).
        kept = [
            t for t in traces if not t.arguments.get("path", "").endswith("chunk_3.md")
        ]
        decoded = exfil.decode(kept)
        assert decoded.count(GAP) == 1
        assert decoded[3] == GAP
        for i, char in enumerate(MESSAGE):
            if i != 3:
                assert decoded[i] == char, (i, decoded)

    def test_two_dropped_turns_two_gaps(self) -> None:
        traces, exfil = _run()
        kept = [
            t
            for t in traces
            if not any(
                t.arguments.get("path", "").endswith(f"chunk_{i}.md") for i in (2, 7)
            )
        ]
        decoded = exfil.decode(kept)
        assert decoded.count(GAP) == 2
        assert decoded[2] == GAP and decoded[7] == GAP

    def test_reordered_turns_unchanged(self) -> None:
        """Position metadata makes decode order-independent."""
        traces, exfil = _run()
        # Swap the write calls of turns 1 and 4.
        by_path = {}
        for t in traces:
            if t.arguments.get("path", "").startswith("chunk_"):
                by_path[t.arguments["path"]] = t
        a, b = by_path["chunk_1.md"], by_path["chunk_4.md"]
        a.arguments["path"], b.arguments["path"] = (
            b.arguments["path"],
            a.arguments["path"],
        )
        a_content = a.arguments["content"]
        a.arguments["content"] = b.arguments["content"]
        b.arguments["content"] = a_content
        assert exfil.decode(traces) == MESSAGE

    def test_corrupt_turn_is_a_gap(self) -> None:
        traces, exfil = _run()
        for t in traces:
            if t.arguments.get("path", "").endswith("chunk_0.md"):
                t.arguments["content"] = "plain text, no frame"
        decoded = exfil.decode(traces)
        assert decoded[0] == GAP
        assert decoded[1:] == MESSAGE[1:]


class TestDetection:
    def test_diversity_blind_with_decoys(self) -> None:
        """Per-turn decoy reads match the clean read-write-read shape."""
        clean = clean_traces(24)
        stego = multiturn_traces(6)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(by_name["calibrated:unique_tools"], stego, clean)
        assert stats["youden_j"] == 0.0

    def test_content_scanner_catches(self) -> None:
        """Honest: each turn's content carries carriers; density catches it."""
        clean = clean_traces(24)
        stego = multiturn_traces(6)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        assert stats["youden_j"] == 1.0

    def test_corpus_observations_distinct(self) -> None:
        runs = multiturn_traces(3)
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
