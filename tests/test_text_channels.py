"""Tests for Unicode/tokenizer-level text channels."""

import pytest

from src.token_stego.text_channels import (
    CHANNEL_REGISTRY,
    HomoglyphChannel,
    InsufficientCoverError,
    NbspChannel,
    NfdChannel,
    TagCharChannel,
    VariationSelectorChannel,
    ZeroWidthChannel,
    all_channels,
    exhaustive_decode,
)

PAYLOAD = b'{"color":"green","number":73,"shape":"square"}'

# Bit-oriented channels carry one bit per carrier position; the framed payload
# is 48 header bits + 45*8 payload bits = 408 bits, so covers must be long.
COVER = (
    "The config file notes that the data report shows the source code and "
    "the test suite for the project. Each line contains the expected value."
)
COVER_LONG = COVER * 4  # ~800 chars
HOMOGLYPH_COVER = "space ace case pace safe ease peas " * 40  # ~1100 eligible letters
SPACE_COVER = "word " * 500  # 500 space positions
ACCENT_COVER = "café résumé naïve élève déjà coûté étudié " * 50  # ~450 accents

COVERS = {
    "zero_width": COVER_LONG,
    "homoglyph": HOMOGLYPH_COVER,
    "tag_char": COVER,
    "nbsp": SPACE_COVER,
    "nfd": ACCENT_COVER,
    "variation_selector": COVER_LONG,
    "case": COVER_LONG,
    "whitespace": SPACE_COVER,
    "heading_level": "## Section A\nbody text here.\n" * 300,
    "word_initial_case": "word " * 500,
}


class TestBitHelpers:
    def test_round_trip_bits(self) -> None:
        from src.token_stego.text_channels import bits_to_bytes, bytes_to_bits

        assert bits_to_bytes(bytes_to_bits(PAYLOAD)) == PAYLOAD

    def test_incomplete_trailing_byte_dropped(self) -> None:
        from src.token_stego.text_channels import bits_to_bytes

        assert bits_to_bytes("1010101") == b""


class TestFraming:
    def test_frame_unframe(self) -> None:
        from src.token_stego.text_channels import _frame_payload_bits, _unframe_bits

        bits = _frame_payload_bits(PAYLOAD)
        assert _unframe_bits(bits) == PAYLOAD

    def test_bad_magic_rejected(self) -> None:
        from src.token_stego.text_channels import _frame_payload_bits, _unframe_bits

        bits = _frame_payload_bits(PAYLOAD)
        flipped = ("1" if bits[0] == "0" else "0") + bits[1:]
        assert _unframe_bits(flipped) is None

    def test_truncated_length_rejected(self) -> None:
        from src.token_stego.text_channels import _frame_payload_bits, _unframe_bits

        bits = _frame_payload_bits(PAYLOAD)
        assert _unframe_bits(bits[: 8 * 8]) is None  # magic only

    def test_trailing_bits_ignored(self) -> None:
        from src.token_stego.text_channels import _frame_payload_bits, _unframe_bits

        bits = _frame_payload_bits(PAYLOAD) + "101010"
        assert _unframe_bits(bits) == PAYLOAD


class TestZeroWidthChannel:
    def test_round_trip(self) -> None:
        ch = ZeroWidthChannel()
        text = ch.encode(PAYLOAD, COVER_LONG)
        assert ch.decode(text) == PAYLOAD

    def test_decode_bits(self) -> None:
        ch = ZeroWidthChannel()
        text = ch.encode_bits("1011", "abcd")
        assert ch.decode_bits(text) == "1011"

    def test_insufficient_cover(self) -> None:
        ch = ZeroWidthChannel()
        with pytest.raises(InsufficientCoverError):
            ch.encode_bits("1" * 100, "short")


class TestHomoglyphChannel:
    def test_round_trip(self) -> None:
        ch = HomoglyphChannel()
        text = ch.encode(PAYLOAD, HOMOGLYPH_COVER)
        assert ch.decode(text) == PAYLOAD

    def test_capacity_counts_eligible_letters(self) -> None:
        ch = HomoglyphChannel()
        assert ch.capacity("cat") == 2  # c and a; t has no lookalike


class TestTagCharChannel:
    def test_round_trip(self) -> None:
        ch = TagCharChannel()
        text = ch.encode(PAYLOAD, COVER)
        assert ch.decode(text) == PAYLOAD

    def test_renders_invisible(self) -> None:
        # All carrier characters are in the tag range; stripping them leaves
        # exactly the cover text.
        ch = TagCharChannel()
        text = ch.encode(PAYLOAD, COVER)
        stripped = "".join(c for c in text if ord(c) < 0xE0000)
        assert stripped == COVER

    def test_non_ascii_payload_rejected(self) -> None:
        ch = TagCharChannel()
        with pytest.raises(ValueError):
            ch.encode(b"\xc3\xa9", COVER)  # UTF-8 é has bytes above 0x7F

    def test_low_bytes_round_trip(self) -> None:
        # Control bytes (e.g. the zero in the length field) are legal carriers.
        ch = TagCharChannel()
        payload = bytes(range(0x00, 0x40))
        assert ch.decode(ch.encode(payload, COVER)) == payload


class TestNbspChannel:
    def test_round_trip(self) -> None:
        ch = NbspChannel()
        text = ch.encode(PAYLOAD, SPACE_COVER)
        assert ch.decode(text) == PAYLOAD

    def test_normalization_destroys_channel(self) -> None:
        ch = NbspChannel()
        text = ch.encode(PAYLOAD, SPACE_COVER)
        normalized = text.replace("\u00a0", " ")
        assert ch.decode(normalized) is None


class TestNfdChannel:
    def test_round_trip(self) -> None:
        ch = NfdChannel()
        text = ch.encode(PAYLOAD, ACCENT_COVER)
        assert ch.decode(text) == PAYLOAD

    def test_nfc_normalization_destroys_channel(self) -> None:
        import unicodedata

        ch = NfdChannel()
        text = ch.encode(PAYLOAD, ACCENT_COVER)
        normalized = unicodedata.normalize("NFC", text)
        assert ch.decode(normalized) is None

    def test_already_decomposed_cover_rejected(self) -> None:
        ch = NfdChannel()
        with pytest.raises(ValueError):
            ch.encode(PAYLOAD, "cafe\u0301")  # decomposed é in cover


class TestVariationSelectorChannel:
    def test_round_trip(self) -> None:
        ch = VariationSelectorChannel()
        text = ch.encode(PAYLOAD, COVER_LONG)
        assert ch.decode(text) == PAYLOAD


class TestRegistry:
    def test_all_channels_round_trip(self) -> None:
        for channel in all_channels():
            text = channel.encode(PAYLOAD, COVERS[channel.name])
            assert channel.decode(text) == PAYLOAD, channel.name

    def test_registry_has_all_channels(self) -> None:
        assert set(CHANNEL_REGISTRY) == {
            "zero_width",
            "homoglyph",
            "tag_char",
            "nbsp",
            "nfd",
            "variation_selector",
            "case",
            "whitespace",
            "heading_level",
            "word_initial_case",
        }

    def test_exhaustive_decode_clean_cover_is_none(self) -> None:
        assert exhaustive_decode(COVER_LONG) is None

    def test_exhaustive_decode_finds_framed_payload(self) -> None:
        ch = HomoglyphChannel()
        text = ch.encode(PAYLOAD, HOMOGLYPH_COVER)
        assert exhaustive_decode(text) == PAYLOAD

    def test_exhaustive_decode_no_false_positive_on_random_text(self) -> None:
        import random

        rng = random.Random(7)
        blob = "".join(rng.choice("abc XYZ0123") for _ in range(500))
        assert exhaustive_decode(blob) is None
