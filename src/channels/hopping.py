"""Seeded multi-carrier channel hopping for text/document carriers.

A coordination-primitive channel: the sender splits the payload into chunks
and encodes each chunk into a separate cover document using a carrier chosen
by a seed-derived schedule. Two seed-derived transforms protect the stream:

1. **Rotation**: the framed bit stream is rotated by a seed-derived offset
   before chunking, so no single chunk document contains the frame magic.
   A per-document exhaustive monitor therefore sees no frame at all — the
   naive "scan each document" monitor is blind by construction.
2. **Schedule**: each chunk is encoded with a different carrier; reassembly
   requires knowing the carrier per chunk.

The receiver, holding the seed, replays both transforms in O(n * c) work. An
unscheduled monitor faces a search over c^n schedules (c carriers, n chunks)
combined with n possible rotations; the frame's magic pins only the true
candidate, and every other valid-looking frame is an ambiguous decoy
(zero-filled payloads pass the magic + length check). Search cost is
therefore exponential in the number of chunks.

Honest negatives, measured in the tests and the evaluation driver:

- Statistical density scanners still flag the channel: every chunk document
  carries a non-zero carrier presence, and a full-registry scanner (OR over
  all carriers) catches every run. Hopping is a *coordination* mechanism, not
  an invisibility mechanism; its per-carrier dilution is real but partial
  (a single-carrier density detector catches only the runs that hop through
  that carrier).
- Brute-force recovery is ambiguous: several schedules/rotations produce
  valid-looking frames, and only one carries the true payload.

The seed is the channel key, playing the same coordination role as the
bootstrapped private languages but for carrier choice and rotation.
"""

from __future__ import annotations

import math
import random

from ..token_stego.text_channels import (
    HomoglyphChannel,
    NbspChannel,
    NfdChannel,
    VariationSelectorChannel,
    ZeroWidthChannel,
    _frame_payload,
    _unframe_bits,
    bytes_to_bits,
)
from .json_order import JsonKeyOrderChannel

CARRIERS = (
    "zero_width",
    "nbsp",
    "nfd",
    "homoglyph",
    "tag_char",
    "variation_selector",
    "json_key_order",
)

_BIT_CARRIERS = {
    "zero_width": ZeroWidthChannel(),
    "nbsp": NbspChannel(),
    "nfd": NfdChannel(),
    "homoglyph": HomoglyphChannel(),
    "variation_selector": VariationSelectorChannel(),
}

_TAG_BASE = 0xE0000
_FRAME_BITS = 48  # magic (32) + uint16 length (16)


def schedule(
    seed: int, n_chunks: int, carriers: tuple[str, ...] = CARRIERS
) -> list[str]:
    """Deterministic per-chunk carrier list derived from the shared seed."""
    rng = random.Random(seed)
    return [rng.choice(carriers) for _ in range(n_chunks)]


def rotation(seed: int, total_bits: int) -> int:
    """Seed-derived rotation offset in [0, total_bits)."""
    return random.Random(seed * 0x9E3779B9 + 1).randrange(0, max(1, total_bits))


def search_space(
    n_chunks: int, total_bits: int, carriers: tuple[str, ...] = CARRIERS
) -> int:
    """Candidate (schedule, rotation) pairs an unscheduled monitor must search."""
    return len(carriers) ** n_chunks * max(1, total_bits)


def carrier_capacity(name: str, cover: str) -> int:
    """Bits one chunk can carry in this cover under this carrier."""
    if name == "json_key_order":
        return JsonKeyOrderChannel.capacity(cover)
    if name == "tag_char":
        # Nibble encoding: one tag char per nibble, one nibble per 4 bits.
        return len(cover) * 4
    return _BIT_CARRIERS[name].capacity(cover)


def carrier_encode_bits(name: str, bits: str, cover: str) -> str:
    """Encode a bit chunk into a cover document with the named carrier."""
    if name == "json_key_order":
        return JsonKeyOrderChannel.encode_bits(bits, cover)
    if name == "tag_char":
        # The tag block mirrors 7-bit values only, so each byte is carried as
        # two tag characters (high nibble, low nibble).
        padded = bits + "0" * ((-len(bits)) % 8)
        payload = int(padded, 2).to_bytes(len(padded) // 8, "big")
        tags: list[str] = []
        for byte in payload:
            tags.append(chr(_TAG_BASE + (byte >> 4)))
            tags.append(chr(_TAG_BASE + (byte & 0x0F)))
        return cover + "".join(tags)
    return _BIT_CARRIERS[name].encode_bits(bits, cover)


def carrier_decode_bits(name: str, text: str) -> str:
    """Recover every carrier bit from a document (may exceed the chunk size)."""
    if name == "json_key_order":
        return JsonKeyOrderChannel.decode_bits(text)
    if name == "tag_char":
        nibbles = [
            ord(c) - _TAG_BASE for c in text if _TAG_BASE <= ord(c) <= _TAG_BASE + 0x0F
        ]
        # Pairs of nibbles form bytes; an unpaired trailing nibble is dropped.
        bytes_ = bytearray(
            (nibbles[i] << 4) | nibbles[i + 1] for i in range(0, len(nibbles) - 1, 2)
        )
        return "".join(format(b, "08b") for b in bytes_)
    return _BIT_CARRIERS[name].decode_bits(text)


class HoppingChannel:
    """Frames a payload once, rotates it, chunks it, and hops carriers."""

    def __init__(
        self, chunk_size: int = 48, carriers: tuple[str, ...] = CARRIERS
    ) -> None:
        if chunk_size <= 0 or chunk_size % 8 != 0:
            raise ValueError(
                "chunk_size must be a positive multiple of 8 (tag_char is byte-oriented)"
            )
        self.chunk_size = chunk_size
        self.carriers = carriers

    def n_chunks(self, num_bits: int) -> int:
        return math.ceil((num_bits + _FRAME_BITS) / self.chunk_size)

    def encode(self, message: str, covers: list[str], seed: int) -> list[str]:
        """Return one stego document per chunk."""
        bits = bytes_to_bits(_frame_payload(message.encode("ascii")))
        rot = rotation(seed, len(bits))
        bits = bits[rot:] + bits[:rot]
        n = self.n_chunks(len(bits) - _FRAME_BITS)
        if len(covers) < n:
            raise ValueError(
                f"hopping needs {n} covers for {n} chunks, got {len(covers)}"
            )
        sched = schedule(seed, n, self.carriers)
        docs: list[str] = []
        for i in range(n):
            chunk = bits[i * self.chunk_size : (i + 1) * self.chunk_size]
            carrier = sched[i]
            cap = carrier_capacity(carrier, covers[i])
            if len(chunk) > cap:
                raise ValueError(
                    f"chunk {i} needs {len(chunk)} bits but {carrier} cover carries {cap}"
                )
            docs.append(carrier_encode_bits(carrier, chunk, covers[i]))
        return docs

    def decode(self, docs: list[str], seed: int, num_bits: int) -> bytes | None:
        """Reassemble the payload using the seed-derived schedule and rotation."""
        n = self.n_chunks(num_bits)
        if len(docs) < n:
            return None
        sched = schedule(seed, n, self.carriers)
        total = num_bits + _FRAME_BITS
        bits: list[str] = []
        for i in range(n):
            chunk_len = min(self.chunk_size, total - i * self.chunk_size)
            raw = carrier_decode_bits(sched[i], docs[i])
            bits.append(raw[:chunk_len])
        stream = "".join(bits)
        rot = rotation(seed, total)
        stream = stream[len(stream) - rot :] + stream[: len(stream) - rot]
        return _unframe_bits(stream)


def brute_force_schedules(
    docs: list[str],
    num_bits: int,
    carriers: tuple[str, ...] = CARRIERS,
    chunk_size: int = 48,
) -> tuple[list[tuple[list[str], int, bytes]], int]:
    """Search all (schedule, rotation) pairs for every valid-looking frame.

    Returns (candidates, attempts) where each candidate is
    (schedule, rotation, payload). Ambiguity is the point: zero-filled chunks
    pass the magic + length check, so several candidates look valid and only
    one carries the true payload. attempts is the total search space explored.
    """
    n = math.ceil((num_bits + _FRAME_BITS) / chunk_size)
    total = num_bits + _FRAME_BITS
    attempts = 0
    candidates: list[tuple[list[str], int, bytes]] = []
    for candidate in _iter_schedules(carriers, n):
        bits: list[str] = []
        for i in range(n):
            chunk_len = min(chunk_size, total - i * chunk_size)
            raw = carrier_decode_bits(candidate[i], docs[i])
            bits.append(raw[:chunk_len])
        stream = "".join(bits)
        for rot in range(total):
            attempts += 1
            unrotated = stream[len(stream) - rot :] + stream[: len(stream) - rot]
            payload = _unframe_bits(unrotated)
            if payload is not None:
                candidates.append((list(candidate), rot, payload))
    return candidates, attempts


def _iter_schedules(carriers: tuple[str, ...], n: int):
    if n == 0:
        yield ()
        return
    for prefix in _iter_schedules(carriers, n - 1):
        for carrier in carriers:
            yield prefix + (carrier,)
