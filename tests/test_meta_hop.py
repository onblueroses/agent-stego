"""Tests for the carrier meta-channel (keyless cross-surface hopping)."""

import json

from src.channels.corpus import (
    _ACCENTED_COVERS,
    _LETTER_RICH_COVERS,
    _PLAIN_COVERS,
    _SPACE_RICH_COVERS,
    _cover_config,
    clean_traces,
    meta_hop_traces,
)
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.meta_hop import (
    HEADER_MAGIC,
    MetaHopChannel,
    MetaHopReportChannel,
)
from src.channels.text_composed import ReportWithDecoysExecutor
from src.token_stego.text_channels import exhaustive_decode

PAYLOAD = b"HELLO"


def _covers_for(carrier: str, i: int) -> str:
    pools = {
        "zero_width": _PLAIN_COVERS,
        "variation_selector": _PLAIN_COVERS,
        "tag_char": _PLAIN_COVERS,
        "nbsp": _SPACE_RICH_COVERS,
        "nfd": _ACCENTED_COVERS,
        "homoglyph": _LETTER_RICH_COVERS,
    }
    return pools[carrier][i % len(pools[carrier])]


def _docs(channel: MetaHopChannel, carriers: list[str], i: int = 0) -> list[str]:
    covers = [_covers_for(c, i) for c in carriers]
    return channel.encode(PAYLOAD.decode(), _cover_config(i), covers, carriers)


class TestMetaHopChannel:
    def test_round_trip(self) -> None:
        channel = MetaHopChannel()
        carriers = ["zero_width", "nbsp"]
        assert channel.n_payload_chunks(40) == len(carriers)
        assert channel.decode(_docs(channel, carriers)) == PAYLOAD

    def test_keyless_schedule(self) -> None:
        """The schedule travels in chunk 0: the receiver needs no seed."""
        channel = MetaHopChannel()
        docs = _docs(channel, ["zero_width", "nbsp"])
        header = channel.extract_header(docs)
        assert header is not None
        assert header.startswith(HEADER_MAGIC)
        assert channel.decode(docs) == PAYLOAD

    def test_text_only_monitor_never_recovers_payload(self) -> None:
        """Insertion carriers are fully blind; substitution carriers on chunk 1
        leak ambiguous garbage frames — never the true payload."""
        channel = MetaHopChannel()
        docs = _docs(channel, ["zero_width", "nbsp"])
        for doc in docs[1:]:
            assert exhaustive_decode(doc) != PAYLOAD

    def test_substitution_chunk1_leaks_ambiguous_frame(self) -> None:
        """Chunk 1 carries magic+length; on a substitution carrier the trailing
        zero-filled positions complete a garbage frame (round-4 ambiguity)."""
        channel = MetaHopChannel()
        docs = _docs(channel, ["homoglyph", "nbsp"])
        found = exhaustive_decode(docs[1])  # chunk 1: homoglyph
        assert found is not None
        assert found != PAYLOAD  # ambiguous garbage, not the message

    def test_structural_only_monitor_gets_schedule_not_payload(self) -> None:
        channel = MetaHopChannel()
        docs = _docs(channel, ["zero_width", "nbsp"])
        header = channel.extract_header(docs)
        assert header is not None
        assert channel.decode([docs[0]]) is None  # schedule alone: no payload

    def test_corrupted_header_fails_closed(self) -> None:
        channel = MetaHopChannel()
        docs = _docs(channel, ["zero_width", "nbsp"])
        from src.channels.json_order import canonicalize

        docs[0] = canonicalize(docs[0])  # canonicalizing monitor kills the header
        assert channel.decode(docs) is None

    def test_wrong_chunk_count_fails(self) -> None:
        channel = MetaHopChannel()
        docs = _docs(channel, ["zero_width", "nbsp"])
        assert channel.decode(docs[:1]) is None


class TestComposition:
    def test_round_trip_via_executor(self) -> None:
        channel = MetaHopReportChannel(
            header_cover=_cover_config(0),
            chunk_covers=[_covers_for("zero_width", 0), _covers_for("nbsp", 0)],
            carriers=["zero_width", "nbsp"],
        )
        result = ReportWithDecoysExecutor().run(channel.encode("HELLO"))
        assert channel.decode(result.traces) == "HELLO"


class TestDetection:
    def test_full_registry_catches(self) -> None:
        """Honest: header and chunks leave per-surface signatures."""
        clean = clean_traces(24)
        stego = meta_hop_traces(8)
        detectors = build_detector_registry(clean)
        key_order = evaluate_detector(
            {d.name: d for d in detectors}["calibrated:key_order_anomaly_fraction"],
            stego,
            clean,
        )
        assert key_order["youden_j"] == 1.0

    def test_observations_distinct(self) -> None:
        runs = meta_hop_traces(3)
        keys = [
            tuple(
                (t.tool_name, json_sorted(t.arguments), round(t.timestamp, 2))
                for t in run
            )
            for run in runs
        ]
        assert len(set(keys)) == len(keys)


def json_sorted(obj: object) -> str:
    return json.dumps(obj, sort_keys=True)
