"""Unicode and tokenizer-level text channels (stdlib only, no model required).

These carriers hide bits in characters that render (nearly) identically to
humans, or that specific model tokenizers delete or normalize. They are a
distinct mechanism family from token-likelihood arithmetic coding: instead of
steering the model's token choice, the sender modifies the *text surface* and
the receiver recovers bits by inspecting characters.

All channels are deterministic text transforms with the same interface:

    channel.encode_bits(bits, cover) -> str      # raise InsufficientCoverError
    channel.decode_bits(text) -> str             # all carried bits, left to right
    channel.encode(payload, cover) -> str        # framed: magic || len || payload
    channel.decode(text) -> bytes | None         # None unless framing verifies

The framed encode embeds a self-describing header (4-byte magic + uint16 BE
length), so decode() recovers the payload without prior knowledge of its
length, and a monitor that guesses the wrong channel almost always sees a
magic mismatch instead of a false positive.

Cover-text contract: cover must not already contain the channel's carrier
characters, otherwise those positions would be misread as payload bits.
encode_bits/encode reject such covers with InsufficientCoverError.

Mechanism sources: these are documented text-steganography carriers (zero-width
characters, Cyrillic homoglyphs, Unicode tag characters, NBSP, composed vs
decomposed accents, variation selectors). The public discussion in
elder-plinius/GLOSSOPETRAE (2026-06) reports which frontier tokenizers strip
or preserve each class; that asymmetry is what makes a channel deliverable to
one model while remaining invisible to a differently-tokenized monitor. The
simulation tables for that asymmetry live in tripartite.py, not here.
"""

from __future__ import annotations

import struct
import unicodedata
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------


def bytes_to_bits(payload: bytes) -> str:
    """Return the 8-bit-per-byte bitstring of a payload."""
    return "".join(format(b, "08b") for b in payload)


def bits_to_bytes(bits: str) -> bytes:
    """Convert a bitstring to bytes, dropping an incomplete trailing byte."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        out.append(int(bits[i : i + 8], 2))
    return bytes(out)


class InsufficientCoverError(ValueError):
    """The cover text cannot carry the requested number of bits."""


# ---------------------------------------------------------------------------
# Self-describing framing (magic || uint16 length || payload)
# ---------------------------------------------------------------------------

# Magic chosen to be unlikely in real carrier text and shared across text
# channels so an exhaustive monitor can try every channel cheaply.
FRAME_MAGIC = b"STX1"


def _frame_payload(payload: bytes, magic: bytes = FRAME_MAGIC) -> bytes:
    """Build magic || uint16 length || payload."""
    return magic + struct.pack(">H", len(payload)) + payload


def _unframe_bytes(framed: bytes, magic: bytes = FRAME_MAGIC) -> bytes | None:
    """Recover the payload from the leading bytes of a framed stream.

    Returns None when the magic does not match or the declared length exceeds
    the available bytes. Trailing bytes beyond the declared payload are ignored
    (they may be cover positions the channel also carries).
    """
    if len(framed) < len(magic) + 2:
        return None
    if framed[: len(magic)] != magic:
        return None
    (length,) = struct.unpack(">H", framed[len(magic) : len(magic) + 2])
    payload_start = len(magic) + 2
    if payload_start + length > len(framed):
        return None
    return framed[payload_start : payload_start + length]


def _frame_payload_bits(payload: bytes, magic: bytes = FRAME_MAGIC) -> str:
    return bytes_to_bits(_frame_payload(payload, magic))


def _unframe_bits(bits: str, magic: bytes = FRAME_MAGIC) -> bytes | None:
    """Recover the payload from the leading bits of a decoded bitstring."""
    return _unframe_bytes(bits_to_bytes(bits), magic)


# ---------------------------------------------------------------------------
# Channel base
# ---------------------------------------------------------------------------


class TextChannel(ABC):
    """A deterministic bit-in-text carrier."""

    name: str = "text"
    #: Characters this channel treats as carriers; covers containing any of
    #: them are rejected so round trips stay exact.
    carrier_chars: frozenset[str] = frozenset()

    @abstractmethod
    def capacity(self, cover: str) -> int:
        """Maximum number of bits this cover can carry."""

    @abstractmethod
    def encode_bits(self, bits: str, cover: str) -> str:
        """Embed bits into cover. Raises InsufficientCoverError when short."""

    @abstractmethod
    def decode_bits(self, text: str) -> str:
        """Return every carried bit observed in text, left to right."""

    def encode(self, payload: bytes, cover: str) -> str:
        """Embed a framed payload (magic + length + payload) into cover."""
        self._check_cover(cover)
        return self.encode_bits(_frame_payload_bits(payload), cover)

    def decode(self, text: str) -> bytes | None:
        """Recover a framed payload, or None when framing does not verify."""
        return _unframe_bits(self.decode_bits(text))

    def _check_cover(self, cover: str) -> None:
        if not self.carrier_chars:
            return
        bad = sorted(set(cover) & self.carrier_chars)
        if bad:
            raise ValueError(
                f"{self.name} cover contains carrier characters {bad!r}; "
                "strip them first or round trips are not exact"
            )

    def _require_capacity(self, cover: str, num_bits: int) -> None:
        cap = self.capacity(cover)
        if num_bits > cap:
            raise InsufficientCoverError(
                f"{self.name} needs {num_bits} bits but cover carries {cap}"
            )


# ---------------------------------------------------------------------------
# Zero-width characters: ZWSP = 0, ZWNJ = 1
# ---------------------------------------------------------------------------


class ZeroWidthChannel(TextChannel):
    """One bit per cover position, encoded as an interleaved invisible char.

    ZWSP (U+200B) = 0, ZWNJ (U+200C) = 1. The encoded text renders with no
    visible change; tokenizers that strip zero-width characters destroy the
    channel for the observer.
    """

    name = "zero_width"
    ZWSP = "\u200b"
    ZWNJ = "\u200c"
    carrier_chars = frozenset(ZWSP + ZWNJ + "\u200d")  # ZWJ too: never a carrier

    def capacity(self, cover: str) -> int:
        return len(cover)

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        out: list[str] = []
        for i, char in enumerate(cover):
            out.append(char)
            if i < len(bits):
                out.append(self.ZWSP if bits[i] == "0" else self.ZWNJ)
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for char in text:
            if char == self.ZWSP:
                bits.append("0")
            elif char == self.ZWNJ:
                bits.append("1")
        return "".join(bits)


# ---------------------------------------------------------------------------
# Homoglyph substitution: Cyrillic lookalikes = 1, Latin originals = 0
# ---------------------------------------------------------------------------


class HomoglyphChannel(TextChannel):
    """One bit per eligible Latin letter, carried by a Cyrillic lookalike.

    Substituting the Cyrillic version encodes 1; leaving the Latin original
    encodes 0. The text renders identically to most readers. A canonicalizing
    monitor that maps the Cyrillic forms back to Latin reads the same bits,
    so this channel is not tokenizer-blindness-resistant; its value is that
    it survives any tokenizer (it is ordinary Unicode text).
    """

    name = "homoglyph"
    # Latin lower -> Cyrillic lookalike (same visual shape), upper -> upper.
    PAIRS: dict[str, str] = {
        "a": "\u0430",  # а
        "c": "\u0441",  # с
        "e": "\u0435",  # е
        "o": "\u043e",  # о
        "p": "\u0440",  # р
        "s": "\u0455",  # ѕ
        "x": "\u0445",  # х
        "y": "\u0443",  # у
    }
    carrier_chars = frozenset(PAIRS.values())

    @staticmethod
    def _latin_lower(char: str) -> str | None:
        return HomoglyphChannel.PAIRS.get(char.lower()) if char.isalpha() else None

    def capacity(self, cover: str) -> int:
        return sum(1 for c in cover if c.isalpha() and c.lower() in self.PAIRS)

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        out: list[str] = []
        bit_idx = 0
        for char in cover:
            cyr = self._latin_lower(char)
            if cyr is None:
                out.append(char)
                continue
            if bit_idx < len(bits) and bits[bit_idx] == "1":
                out.append(cyr.upper() if char.isupper() else cyr)
            else:
                out.append(char)
            bit_idx += 1
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for char in text:
            lower = char.lower()
            if lower in self.PAIRS.values():
                bits.append("1")
            elif lower in self.PAIRS:
                bits.append("0")
        return "".join(bits)


# ---------------------------------------------------------------------------
# Unicode tag characters: U+E0000 + ASCII code point (invisible mirror)
# ---------------------------------------------------------------------------


class TagCharChannel(TextChannel):
    """Payload carried as invisible Unicode tag characters.

    Each payload byte b (printable ASCII recommended) becomes the tag
    character U+E0000 + b, appended after the cover text. Tag characters
    render as nothing; some tokenizers strip them entirely while others
    preserve them, which is the asymmetry the tripartite harness exploits.

    This channel is payload-oriented (bits are carried 8 at a time as byte
    values), so encode_bits is not a sensible entry point and raises.
    """

    name = "tag_char"
    TAG_BASE = 0xE0000
    TAG_MIN = 0xE0020  # space is the smallest useful tag character
    TAG_MAX = 0xE007E  # ~ (tilde)
    carrier_chars = frozenset(chr(cp) for cp in range(0xE0000, 0xE0080))

    def capacity(self, cover: str) -> int:
        # Each cover position can host one tag character = one payload byte.
        return len(cover) * 8

    def encode_bits(self, bits: str, cover: str) -> str:
        raise NotImplementedError(
            f"{self.name} is byte-oriented; use encode(payload, cover)"
        )

    def encode(self, payload: bytes, cover: str) -> str:
        self._check_cover(cover)
        framed = _frame_payload(payload)
        # Every byte 0x00-0x7F maps to an invisible tag character; bytes above
        # 0x7F have no tag mirror and cannot be carried.
        tags = "".join(chr(self.TAG_BASE + b) if b <= 0x7F else "?" for b in framed)
        if "?" in tags:
            raise ValueError("tag_char payload bytes must be in 0x00-0x7F")
        return cover + tags

    def decode_bits(self, text: str) -> str:
        raise NotImplementedError(f"{self.name} is byte-oriented; use decode(text)")

    def decode(self, text: str) -> bytes | None:
        ascii_bytes = bytearray()
        for char in text:
            cp = ord(char)
            if self.TAG_BASE <= cp <= self.TAG_BASE + 0x7F:
                ascii_bytes.append(cp - self.TAG_BASE)
        if not ascii_bytes:
            return None
        return _unframe_bytes(bytes(ascii_bytes))


# ---------------------------------------------------------------------------
# NBSP: regular space = 0, non-breaking space = 1
# ---------------------------------------------------------------------------


class NbspChannel(TextChannel):
    """One bit per space position, carried by NBSP (U+00A0) vs space.

    Most readers cannot tell the two apart. Tokenizers differ in whether they
    normalize NBSP to a regular space, which makes the channel visible to one
    observer and invisible to another.
    """

    name = "nbsp"
    NBSP = "\u00a0"
    carrier_chars = frozenset(NBSP)

    def capacity(self, cover: str) -> int:
        return cover.count(" ")

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        out: list[str] = []
        bit_idx = 0
        for char in cover:
            if char == " " and bit_idx < len(bits):
                out.append(self.NBSP if bits[bit_idx] == "1" else " ")
                bit_idx += 1
            else:
                out.append(char)
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for char in text:
            if char == self.NBSP:
                bits.append("1")
            elif char == " ":
                bits.append("0")
        return "".join(bits)


# ---------------------------------------------------------------------------
# NFD: composed accent = 0, decomposed (base + combining mark) = 1
# ---------------------------------------------------------------------------

#: Precomposed characters that decompose into base + a single combining mark,
#: which we use as carriers. Built from the Latin-1 supplement and Latin
#: Extended-A blocks.
_DECOMPOSABLE: dict[str, str] = {}
for _cp in range(0x00C0, 0x0250):
    _char = chr(_cp)
    _nfd = unicodedata.normalize("NFD", _char)
    if len(_nfd) == 2 and _nfd != _char and unicodedata.combining(_nfd[1]):
        _DECOMPOSABLE[_char] = _nfd
_DECOMPOSABLE_BASES: frozenset[str] = frozenset(
    nfd[0] for nfd in _DECOMPOSABLE.values()
)
_DECOMPOSABLE_MARKS: frozenset[str] = frozenset(
    nfd[1] for nfd in _DECOMPOSABLE.values()
)


class NfdChannel(TextChannel):
    """One bit per decomposable accent, carried by NFC vs NFD form.

    Composed form (e.g. é, U+00E9) encodes 0; the decomposed form (e +
    combining acute, U+0065 U+0301) encodes 1. The two render identically.
    Tokenizers that normalize to NFC erase the channel; those that preserve
    the input keep it. This is the stealthiest of the carriers here because
    even a byte-level comparison of the rendered glyphs shows no difference.
    """

    name = "nfd"
    carrier_chars = (
        frozenset(_DECOMPOSABLE.keys()) | _DECOMPOSABLE_BASES | _DECOMPOSABLE_MARKS
    )

    def _check_cover(self, cover: str) -> None:
        # Unlike the other channels, a composed accented cover is exactly what
        # this channel needs; only already-decomposed input (a base followed
        # by a combining mark) would be misread as payload bits.
        bad = [c for c in cover if c in _DECOMPOSABLE_MARKS]
        if bad:
            raise ValueError(
                f"{self.name} cover already contains combining marks {bad[:5]!r}; "
                "normalize the cover to NFC first or round trips are not exact"
            )

    def capacity(self, cover: str) -> int:
        return sum(1 for c in cover if c in _DECOMPOSABLE)

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        out: list[str] = []
        bit_idx = 0
        for char in cover:
            decomp = _DECOMPOSABLE.get(char)
            if decomp is None:
                out.append(char)
                continue
            if bit_idx < len(bits) and bits[bit_idx] == "1":
                out.append(decomp)
            else:
                out.append(char)
            bit_idx += 1
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        i = 0
        while i < len(text):
            char = text[i]
            if char in _DECOMPOSABLE:
                bits.append("0")
                i += 1
                continue
            # A base followed by a combining mark that jointly form one of our
            # precomposed carriers is a decomposed (bit 1) position.
            if (
                char in _DECOMPOSABLE_BASES
                and i + 1 < len(text)
                and text[i + 1] in _DECOMPOSABLE_MARKS
            ):
                pair = char + text[i + 1]
                if pair in _DECOMPOSABLE.values():
                    bits.append("1")
                    i += 2
                    continue
            i += 1
        return "".join(bits)


# ---------------------------------------------------------------------------
# Variation selectors: VS15 (U+FE0E) = 0, VS16 (U+FE0F) = 1
# ---------------------------------------------------------------------------


class VariationSelectorChannel(TextChannel):
    """One bit per cover position, carried by an appended variation selector.

    VS15 (U+FE0E, text presentation) = 0, VS16 (U+FE0F, emoji presentation) =
    1. Selectors are widely used after emoji and dingbats; a monitor that
    strips them loses the channel. This mirrors the emoji-steganography
    variant (payload in skin-tone / presentation selectors) at the bit level.
    """

    name = "variation_selector"
    VS15 = "\ufe0e"
    VS16 = "\ufe0f"
    carrier_chars = frozenset(VS15 + VS16)

    def capacity(self, cover: str) -> int:
        return len(cover)

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        out: list[str] = []
        for i, char in enumerate(cover):
            out.append(char)
            if i < len(bits):
                out.append(self.VS15 if bits[i] == "0" else self.VS16)
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for char in text:
            if char == self.VS15:
                bits.append("0")
            elif char == self.VS16:
                bits.append("1")
        return "".join(bits)


# ---------------------------------------------------------------------------
# Case channel: bits in letter case (uppercase/lowercase)
# ---------------------------------------------------------------------------


_SENTENCE_END = set(".!?")


def _eligible_case_indices(text: str) -> list[int]:
    """Indices of ASCII letters that are NOT sentence-initial (carry bits).

    A letter is sentence-initial when it starts the text or follows one of
    '.!?' (ignoring whitespace). Both sender and receiver use this exact
    rule, so round trips are deterministic; natural sentence capitals are
    never read as payload.
    """
    out: list[int] = []
    prev_nonspace: str | None = None
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        if ch.isascii() and ch.isalpha():
            if prev_nonspace is not None and prev_nonspace not in _SENTENCE_END:
                out.append(i)
        prev_nonspace = ch
    return out


class CaseChannel(TextChannel):
    """One bit per mid-sentence letter: lowercase = 0, uppercase = 1.

    Case is the most visible typographic property — and yet no density or
    structure scanner in the detector registry models case regularity, so
    an all-ASCII case-flipped cover passes every registered content
    feature. The sender canonicalizes the cover (mid-sentence letters to
    lowercase) before embedding, so the bit pattern is exact; the receiver
    reads the same eligibility rule. Honest negatives: the pattern is
    visible to a human reader and to any case-aware monitor, and the
    canonicalized cover reads typographically flat.
    """

    name = "case"
    carrier_chars = frozenset()  # covers are canonicalized, not rejected

    def capacity(self, cover: str) -> int:
        return len(_eligible_case_indices(cover))

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        chars = list(cover)
        positions = _eligible_case_indices(cover)
        for i, pos in enumerate(positions):
            chars[pos] = chars[pos].lower()
        for i, pos in enumerate(positions[: len(bits)]):
            chars[pos] = chars[pos].upper() if bits[i] == "1" else chars[pos].lower()
        return "".join(chars)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for pos in _eligible_case_indices(text):
            ch = text[pos]
            bits.append("1" if ch.isupper() else "0")
        return "".join(bits)


# ---------------------------------------------------------------------------
# Whitespace channel: bits in single vs double space runs
# ---------------------------------------------------------------------------


def _space_run_ends(text: str) -> list[int]:
    """End indices of pure-space runs (no newline/tab) in text.

    A run is eligible only if it consists solely of ASCII spaces, so
    paragraph breaks (newlines) are never read as payload by either side.
    The end index points past the run.
    """
    out: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == " ":
            j = i
            while j < n and text[j] == " ":
                j += 1
            out.append(j)
            i = j
        else:
            i += 1
    return out


class WhitespaceChannel(TextChannel):
    """One bit per inter-word space run: single space = 0, double = 1.

    Double spaces are the most common typing artifact in real documents, so
    a cover with a few doubled spaces is unremarkable — and no registry
    feature counts regular-space runs (nbsp_density is about U+00A0 only).
    The sender canonicalizes every pure-space run to a single space, then
    doubles the first K runs for bit 1. Honest negatives: every text
    linter flags double spaces, so a lint-style consistency monitor catches
    it trivially — the tell is obvious to anyone who checks.
    """

    name = "whitespace"
    carrier_chars = frozenset()  # runs are canonicalized, not rejected

    def capacity(self, cover: str) -> int:
        return len(_space_run_ends(cover))

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        out: list[str] = []
        cursor = 0
        for i, end in enumerate(_space_run_ends(cover)):
            start = end
            while start > cursor and cover[start - 1] == " ":
                start -= 1
            out.append(cover[cursor:start])
            if i < len(bits):
                out.append("  " if bits[i] == "1" else " ")
            else:
                out.append(" ")
            cursor = end
        out.append(cover[cursor:])
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        cursor = 0
        for end in _space_run_ends(text):
            start = end
            while start > cursor and text[start - 1] == " ":
                start -= 1
            length = end - start
            bits.append("1" if length >= 2 else "0")
            cursor = end
        return "".join(bits)


# ---------------------------------------------------------------------------
# Word-initial case channel: bits in word-initial letter case
# ---------------------------------------------------------------------------


def _word_initial_indices(text: str) -> list[int]:
    """Indices of word-initial letters that are NOT sentence-initial.

    A letter is word-initial when the previous character is whitespace.
    It is additionally SKIPPED (ineligible) when it starts the text or
    follows '.!?' (ignoring whitespace) — a sentence-initial capital is
    natural and never a bit. Both sides share this exact rule.
    """
    out: list[int] = []
    prev: str | None = None
    prev_nonspace: str | None = None
    for i, ch in enumerate(text):
        if ch.isspace():
            prev = ch
            continue
        if ch.isascii() and ch.isalpha():
            if prev is not None and prev.isspace():
                if prev_nonspace is not None and prev_nonspace not in _SENTENCE_END:
                    out.append(i)
        prev = ch
        if not ch.isspace():
            prev_nonspace = ch
    return out


class WordInitialChannel(TextChannel):
    """One bit per eligible word-initial letter: lowercase = 0, uppercase = 1.

    The formatting closure's case sub-score counts only MID-WORD flips
    (uppercase preceded by a lowercase letter), by design, to avoid
    proper-noun false positives. The flip side is a precision gap: a
    channel that flips only WORD-INITIAL letters is invisible to the
    closure and to every other registry feature. The gap is structurally
    unclosable by a syntactic rule: a flipped "Report" is byte-identical
    to a proper noun "Report". The sender canonicalizes eligible
    word-initials to lowercase and the cover must be proper-noun-free;
    the receiver reads uppercase=1 / lowercase=0.
    """

    name = "word_initial_case"
    carrier_chars = frozenset()

    def capacity(self, cover: str) -> int:
        return len(_word_initial_indices(cover))

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        self._require_capacity(cover, len(bits))
        chars = list(cover)
        positions = _word_initial_indices(cover)
        for i, pos in enumerate(positions):
            chars[pos] = chars[pos].lower()
        for i, pos in enumerate(positions[: len(bits)]):
            chars[pos] = chars[pos].upper() if bits[i] == "1" else chars[pos].lower()
        return "".join(chars)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for pos in _word_initial_indices(text):
            bits.append("1" if text[pos].isupper() else "0")
        return "".join(bits)


# ---------------------------------------------------------------------------
# Heading-level channel: bits in markdown heading levels
# ---------------------------------------------------------------------------


_HEADING_RE = __import__("re").compile(
    r"^(#{1,4})\s+", flags=__import__("re").MULTILINE
)


class HeadingChannel(TextChannel):
    """Bits in markdown heading levels.

    The default codebook is 4 levels (#, ##, ###, ####) = 2 bits per
    heading. The SMOOTH mode (`levels=2`) uses only # and ## with
    alternating nesting (# -> ## -> # -> ## is normal document structure),
    so it never triggers the closure's skip rule: it is the heading-side
    precision gap, invisible to the full registry (round 71, GAP-1).

    Long documents — specs, READMEs, changelogs — carry many headings, and
    heading levels legitimately vary with nesting depth. The channel sets
    each heading's level from the bit stream (cover headings rewritten,
    non-heading text untouched).
    """

    name = "heading_level"
    carrier_chars = frozenset()

    def __init__(self, levels: int = 4) -> None:
        if levels not in (2, 4):
            raise ValueError("heading channel supports 2 or 4 levels")
        self.levels = levels
        self.bits_per_heading = 1 if levels == 2 else 2

    def capacity(self, cover: str) -> int:
        return len(_HEADING_RE.findall(cover))

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        bph = self.bits_per_heading
        if len(bits) % bph:
            bits = bits.ljust(len(bits) + (bph - len(bits) % bph), "0")
        headings = list(_HEADING_RE.finditer(cover))
        if len(bits) // bph > len(headings):
            raise InsufficientCoverError(
                f"{self.name} needs {len(bits) // bph} headings but cover has {len(headings)}"
            )
        out: list[str] = []
        cursor = 0
        bit_cursor = 0
        for i, m in enumerate(headings):
            out.append(cover[cursor : m.start()])
            if i < len(bits) // bph:
                level = int(bits[bit_cursor : bit_cursor + bph], 2) + 1
                bit_cursor += bph
            else:
                level = 1
            out.append("#" * level + " ")
            cursor = m.end()
        out.append(cover[cursor:])
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        bph = self.bits_per_heading
        for m in _HEADING_RE.finditer(text):
            level = min(len(m.group(1)), self.levels)
            bits.append(format(level - 1, f"0{bph}b"))
        return "".join(bits)


# ---------------------------------------------------------------------------
# Registry for exhaustive monitors and tooling
# ---------------------------------------------------------------------------

CHANNEL_REGISTRY: dict[str, TextChannel] = {
    ch.name: ch
    for ch in (
        ZeroWidthChannel(),
        HomoglyphChannel(),
        TagCharChannel(),
        NbspChannel(),
        NfdChannel(),
        VariationSelectorChannel(),
        CaseChannel(),
        WhitespaceChannel(),
        HeadingChannel(),
        WordInitialChannel(),
    )
}


def all_channels() -> list[TextChannel]:
    """Return one instance of every text channel, in a stable order."""
    return list(CHANNEL_REGISTRY.values())


def exhaustive_decode(
    text: str, channels: list[TextChannel] | None = None
) -> bytes | None:
    """Try every channel's decode and return the first framed payload.

    This is the deterministic stand-in for a monitor that does not know which
    channel was used: without a valid magic header the payload is not
    recovered, which keeps false positives on clean text at zero.
    """
    for channel in channels or all_channels():
        try:
            payload = channel.decode(text)
        except NotImplementedError:
            continue
        if payload is not None:
            return payload
    return None
