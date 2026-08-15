"""Date-format channel: bits in which locale-valid date format is used.

Changelogs, datelines, and logs legitimately carry many dates, and date
formats legitimately vary by locale (ISO, slash, dot, "14 Aug 2026",
"Aug 14, 2026", compact). The channel uses an 8-format codebook: each date
in a document carries 3 bits by which of the 8 formats renders it. Every
format is a REAL locale convention, so no single date is anomalous — only
the MIX is the payload.

- Content independence: the document is a genuine changelog; the dates are
  real dates; the format mix is the only signal.
- Registry scope: the current detector registry does not model date-format
  consistency. That is a fact about this implementation, not evidence that the
  carrier is generally invisible.
- The codebook (format -> bits) is the shared protocol.

Honest negatives: a locale-consistency monitor (or a careful human) sees a
document mixing date formats; an exhaustive text-channel scan does not
find it (it is not a character carrier); the receiver must know the
codebook.
"""

from __future__ import annotations

import re
from datetime import date

from ..token_stego.text_channels import InsufficientCoverError, TextChannel

#: Eight locale-valid renderings of a date -> 3 bits.
#: strftime-style formats, ordered by index (bit value). Two measured
#: scanner boundaries shape the codebook: dot-dates ("14.08.2026") read as
#: "dd.mm" tokens ending in 0, and hyphen-dates ("2026-08-14") read as
#: "-0"-style negative-number tokens — both trigger the registry's
#: number-format scanner at J=1.0 (see tests). The eight formats below are
#: dot-free AND hyphen-free: slash, month-name, compact — every one is a
#: real locale convention.
DATE_FORMATS: tuple[str, ...] = (
    "%Y/%m/%d",  # 0: 2026/08/14 (ISO slash)
    "%m/%d/%Y",  # 1: 08/14/2026 (US)
    "%d %b %Y",  # 2: 14 Aug 2026
    "%b %d, %Y",  # 3: Aug 14, 2026
    "%b %d %Y",  # 4: Aug 14 2026
    "%Y%m%d",  # 5: 20260814 (compact)
    "%d %B %Y",  # 6: 14 August 2026 (full month)
    "%B %d, %Y",  # 7: August 14, 2026 (full month, comma)
)

#: Classification regexes, one per format index, disjoint by construction
#: (checked in tests): order matters — shorter month-name patterns before
#: their full-month twins, comma format before its no-comma twin.
_FORMAT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d{4}/\d{2}/\d{2}"),
    re.compile(r"\d{2}/\d{2}/\d{4}"),
    re.compile(r"\d{1,2} [A-Za-z]{3} \d{4}"),
    re.compile(r"[A-Za-z]{3} \d{1,2}, \d{4}"),
    re.compile(r"[A-Za-z]{3} \d{1,2} \d{4}"),
    re.compile(r"\d{8}"),
    re.compile(r"\d{1,2} [A-Za-z]{3,} \d{4}"),
    re.compile(r"[A-Za-z]{3,} \d{1,2}, \d{4}"),
)

_MASTER = re.compile("|".join(f"({p.pattern})" for p in _FORMAT_PATTERNS))

#: The cover's base form: hyphenated ISO dates the encoder parses.
_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _count_iso_dates(text: str) -> int:
    """Number of base-form (hyphen ISO) dates in a cover."""
    return len(_ISO_RE.findall(text))


def _iter_dates(text: str) -> list[tuple[int, int]]:
    """Return (start, format_index) for every codebook date found, in text
    order. Used by decode on stego text (which contains only codebook
    formats); covers are counted with _count_iso_dates instead."""
    out: list[tuple[int, int]] = []
    for match in _MASTER.finditer(text):
        for idx, group in enumerate(match.groups()):
            if group is not None:
                out.append((match.start(), idx))
                break
    return out


class DateFormatChannel(TextChannel):
    """One date = 3 bits, via an 8-format codebook.

    The cover must contain dates in ISO form (the channel's base format);
    encode rewrites every date to the codebook format for its 3 bits.
    decode classifies every date by format and unframes.
    """

    name = "date_format"
    carrier_chars = frozenset()  # dates are rewritten, not rejected

    def capacity(self, cover: str) -> int:
        return _count_iso_dates(cover)

    def encode_bits(self, bits: str, cover: str) -> str:
        if any(bit not in "01" for bit in bits):
            raise ValueError("bits must contain only '0' and '1'")
        if len(bits) % 3:
            bits = bits.ljust(len(bits) + (3 - len(bits) % 3), "0")
        n_dates = _count_iso_dates(cover)
        if len(bits) // 3 > n_dates:
            raise InsufficientCoverError(
                f"{self.name} needs {len(bits) // 3} dates but cover has {n_dates}"
            )
        # Collect the ISO dates from the cover, then rebuild with new formats.
        # Every date is rewritten: the first `needed` carry the payload, the
        # rest use format 0 (slash) so no hyphenated date survives.
        iso_parts: list[tuple[int, str]] = [
            (m.start(), m.group(0)) for m in _ISO_RE.finditer(cover)
        ]
        needed = len(bits) // 3
        out: list[str] = []
        cursor = 0
        bit_cursor = 0
        for i, (pos, iso) in enumerate(iso_parts):
            out.append(cover[cursor:pos])
            if i < needed:
                fmt_index = int(bits[bit_cursor : bit_cursor + 3], 2)
                bit_cursor += 3
            else:
                fmt_index = 0
            year, month, day = (int(x) for x in iso.split("-"))
            out.append(date(year, month, day).strftime(DATE_FORMATS[fmt_index]))
            cursor = pos + len(iso)
        out.append(cover[cursor:])
        return "".join(out)

    def decode_bits(self, text: str) -> str:
        bits: list[str] = []
        for _, fmt_index in _iter_dates(text):
            bits.append(format(fmt_index, "03b"))
        return "".join(bits)


# Re-export for downstream imports
__all__ = ["DateFormatChannel", "DATE_FORMATS", "_iter_dates"]
