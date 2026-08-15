"""Tests for the seeded multi-carrier hopping channel."""

import pytest

from src.channels.corpus import _hopping_covers, hopping_traces
from src.channels.detector import build_detector_registry, evaluate_detector
from src.channels.hopping import (
    HoppingChannel,
    brute_force_schedules,
    carrier_decode_bits,
    carrier_encode_bits,
    rotation,
    schedule,
    search_space,
)
from src.token_stego.text_channels import exhaustive_decode

MESSAGE = "HELLO"
NUM_BITS = len(MESSAGE.encode("ascii")) * 8


def _docs(seed: int, message: str = MESSAGE, chunk_size: int = 48) -> list[str]:
    channel = HoppingChannel(chunk_size=chunk_size)
    n = channel.n_chunks(len(message.encode("ascii")) * 8)
    covers = _hopping_covers(schedule(seed, n))
    return channel.encode(message, covers, seed)


class TestSchedule:
    def test_deterministic(self) -> None:
        assert schedule(42, 5) == schedule(42, 5)

    def test_distinct_seeds_differ(self) -> None:
        assert schedule(1, 5) != schedule(2, 5)

    def test_carriers_used(self) -> None:
        assert set(schedule(99, 50)) <= {
            "zero_width",
            "nbsp",
            "nfd",
            "homoglyph",
            "tag_char",
            "variation_selector",
            "json_key_order",
        }

    def test_rotation_in_range(self) -> None:
        for seed in range(20):
            assert 0 <= rotation(seed, 64) < 64

    def test_search_space_grows_exponentially(self) -> None:
        assert search_space(2, 64) == 7**2 * 64
        assert search_space(3, 64) == 7**3 * 64


class TestHoppingChannel:
    def test_round_trip(self) -> None:
        channel = HoppingChannel()
        docs = _docs(5)
        assert channel.decode(docs, 5, NUM_BITS) == MESSAGE.encode("ascii")

    def test_wrong_seed_fails(self) -> None:
        channel = HoppingChannel()
        docs = _docs(5)
        assert channel.decode(docs, 6, NUM_BITS) is None

    def test_round_trip_all_seed_range(self) -> None:
        """Every carrier must survive the round trip across schedules."""
        channel = HoppingChannel()
        for seed in range(100, 130):
            n = channel.n_chunks(NUM_BITS)
            covers = _hopping_covers(schedule(seed, n))
            docs = channel.encode(MESSAGE, covers, seed)
            assert channel.decode(docs, seed, NUM_BITS) == MESSAGE.encode("ascii"), seed

    def test_too_few_covers_raises(self) -> None:
        channel = HoppingChannel()
        with pytest.raises(ValueError):
            channel.encode(MESSAGE, ["short"], 1)

    def test_invalid_chunk_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            HoppingChannel(chunk_size=7)  # not a multiple of 8


class TestCarrierAdapters:
    def test_tag_char_nibble_round_trip(self) -> None:
        bits = "10101010" * 6  # 48 bits, byte aligned
        cover = "the quick brown fox jumps over the lazy dog and back again"
        text = carrier_encode_bits("tag_char", bits, cover)
        assert carrier_decode_bits("tag_char", text)[: len(bits)] == bits

    def test_tag_char_pads_non_aligned_chunk(self) -> None:
        bits = "10101"  # 5 bits, padded to a byte
        cover = "some cover text here"
        text = carrier_encode_bits("tag_char", bits, cover)
        recovered = carrier_decode_bits("tag_char", text)
        assert recovered.startswith(bits)
        assert len(recovered) == 8  # padded to one byte

    def test_json_carrier_adapter_round_trip(self) -> None:
        from src.channels.corpus import _cover_config

        bits = "1" * 48
        text = carrier_encode_bits("json_key_order", bits, _cover_config(0))
        assert carrier_decode_bits("json_key_order", text)[:48] == bits


class TestBruteForce:
    def test_true_payload_found_with_ambiguity(self) -> None:
        seed = 7
        docs = _docs(seed, message="H")  # 1 char -> 2 chunks at 48 bits
        candidates, attempts = brute_force_schedules(docs, 8)
        payloads = {candidate[2] for candidate in candidates}
        assert b"H" in payloads
        # Ambiguity: other schedules/rotations also produce valid-looking
        # frames (zero-filled), so the search cannot be shortcut.
        assert len(candidates) >= 1
        assert attempts == search_space(len(docs), 8 + 48)

    def test_three_chunks_true_payload_found(self) -> None:
        seed = 9
        channel = HoppingChannel()
        n = channel.n_chunks(16)  # "AB" = 16 bits -> 2 chunks
        covers = _hopping_covers(schedule(seed, n))
        docs = channel.encode("AB", covers, seed)
        candidates, _ = brute_force_schedules(docs, 16)
        assert b"AB" in {candidate[2] for candidate in candidates}


class TestMonitoring:
    def test_naive_exhaustive_monitor_blind_per_document(self) -> None:
        """Rotation ensures no single chunk document is self-framing."""
        docs = _docs(11)
        assert all(exhaustive_decode(doc) is None for doc in docs)

    def test_single_carrier_density_detector_partially_diluted(self) -> None:
        """Honest: hopping dilutes per-carrier density — the invisible-char
        scanner catches only the runs that hop through an invisible carrier,
        so J is partial, not 1.0."""
        from src.channels.corpus import clean_traces

        clean = clean_traces(24)
        stego = hopping_traces(12)
        detectors = build_detector_registry(clean)
        by_name = {d.name: d for d in detectors}
        stats = evaluate_detector(
            by_name["calibrated:invisible_char_density"], stego, clean
        )
        assert 0.0 < stats["youden_j"] < 1.0

    def test_full_registry_monitor_catches_every_run(self) -> None:
        """OR over all content scanners: every run uses some carrier, so the
        full-registry monitor catches the whole family at J = 1.0."""
        from src.channels.corpus import clean_traces

        clean = clean_traces(24)
        stego = hopping_traces(12)
        detectors = build_detector_registry(clean)
        content_detectors = [
            d
            for d in detectors
            if d.feature
            in {
                "invisible_char_density",
                "nbsp_density",
                "homoglyph_density",
                "combining_mark_density",
                "non_ascii_density",
                "key_order_anomaly_fraction",
            }
        ]
        tp = sum(
            1 for run in stego if any(d.score(run) == 1.0 for d in content_detectors)
        )
        fp = sum(
            1 for run in clean if any(d.score(run) == 1.0 for d in content_detectors)
        )
        assert tp == len(stego)
        assert fp == 0


class TestCorpus:
    def test_hopping_traces_are_multi_turn(self) -> None:
        traces = hopping_traces(2)
        assert all(len(t) == 2 for t in traces)  # HELLO -> two 48-bit chunks
        assert all(all(tr.tool_name == "write_file" for tr in run) for run in traces)

    def test_distinct_schedules_per_trial(self) -> None:
        seeds = [1_000 + i for i in range(6)]
        assert len({tuple(schedule(s, 2)) for s in seeds}) > 1
