"""Arithmetic coding for token-likelihood steganography.

Encodes hidden message bits into token selections by partitioning probability
intervals. It uses message-scaled `decimal.Decimal` precision and fails closed
when the final interval cannot guarantee the requested bit prefix.

The message bitstring is interpreted as a binary fraction 0.b1b2b3... defining
a target point in [0, 1). At each generation step, the probability distribution
partitions the current interval into token sub-intervals. The encoder selects
the token whose sub-interval contains the target point, narrowing the interval.
The decoder reverses this by narrowing the same interval from observed tokens,
then reads off the binary expansion.

To guarantee exact recovery, the caller must provide enough generation steps
that the final interval lies wholly inside the requested dyadic message cell.
"""

from collections.abc import Sequence
from decimal import Decimal, localcontext

# The pre-audit implementation used a 50-digit floor and 30 guard digits. Keep
# both as the historical replay contract while allocating one exact decimal
# place per dyadic payload bit. Interval-containment checks remain the final
# correctness gate; these values are not a numerical-error proof.
_HISTORICAL_PRECISION_FLOOR = 50
_PARTITION_GUARD_DIGITS = 30

Distribution = list[Decimal]


class InsufficientCapacityError(ValueError):
    """The observed interval does not guarantee the requested message bits."""


def _precision_for_bits(num_bits: int) -> int:
    """Decimal digits needed for exact dyadic message-cell boundaries."""
    return max(_HISTORICAL_PRECISION_FLOOR, num_bits) + _PARTITION_GUARD_DIGITS


def _normalize(dist: Sequence[float | Decimal]) -> Distribution:
    """Convert a distribution to Decimal and normalize so it sums to exactly 1."""
    if not dist:
        raise ValueError("Distribution must contain at least one probability")
    d = [Decimal(str(p)) for p in dist]
    if any(not p.is_finite() or p < 0 for p in d):
        raise ValueError("Distribution probabilities must be finite and non-negative")
    total = sum(d)
    if total <= 0:
        raise ValueError("Distribution must have positive probability mass")
    d = [p / total for p in d]
    residual = Decimal(1) - sum(d)
    correction_index = max(range(len(d)), key=d.__getitem__)
    d[correction_index] += residual
    if d[correction_index] < 0:
        raise ArithmeticError("Distribution normalization produced negative mass")
    return d


def _bits_to_point(bits: str) -> Decimal:
    """Convert a bitstring to a Decimal in [0, 1) via binary fraction.

    To avoid boundary issues, we place the target at the midpoint of the
    message's binary interval [val, val + 2^-len), i.e. val + 2^-(len+1).
    This ensures the target never sits exactly on a sub-interval boundary.
    """
    value = Decimal(0)
    scale = Decimal("0.5")
    for b in bits:
        if b == "1":
            value += scale
        scale /= 2
    # scale is now 2^(-(len+1)); add it to reach midpoint
    value += scale
    return value


def _point_to_bits(value: Decimal, num_bits: int) -> str:
    """Convert a Decimal in [0, 1) to a bitstring of given length."""
    bits: list[str] = []
    remainder = value
    threshold = Decimal("0.5")
    for _ in range(num_bits):
        if remainder >= threshold:
            bits.append("1")
            remainder -= threshold
        else:
            bits.append("0")
        threshold /= 2
    return "".join(bits)


def _partition_step(
    low: Decimal,
    high: Decimal,
    probs: Distribution,
) -> list[tuple[Decimal, Decimal]]:
    """Partition [low, high) into sub-intervals proportional to probs."""
    width = high - low
    n = len(probs)
    intervals: list[tuple[Decimal, Decimal]] = []
    cum_lo = low
    for i in range(n):
        if i == n - 1:
            cum_hi = high  # avoid rounding gap
        else:
            cum_hi = cum_lo + width * probs[i]
        intervals.append((cum_lo, cum_hi))
        cum_lo = cum_hi
    return intervals


class ArithmeticEncoder:
    """Encodes message bits into token index selections."""

    def __init__(self, message_bits: str) -> None:
        if any(bit not in "01" for bit in message_bits):
            raise ValueError("Message must contain only '0' and '1' bits")
        self.message_bits = message_bits
        self._precision = _precision_for_bits(len(message_bits))
        with localcontext() as ctx:
            ctx.prec = self._precision
            self.target = _bits_to_point(message_bits)
            self.low = Decimal(0)
            self.high = Decimal(1)
            self._committed_bits = 0

    @property
    def bits_consumed(self) -> int:
        """Number of leading message bits guaranteed by the current interval.

        Width alone is not sufficient: an interval narrower than a bit cell can
        still straddle that cell's boundary. A bit is committed only when the
        entire coding interval lies inside the corresponding dyadic prefix cell.
        """
        with localcontext() as ctx:
            ctx.prec = self._precision
            while self._committed_bits < len(self.message_bits):
                prefix_len = self._committed_bits + 1
                denominator = Decimal(2) ** prefix_len
                prefix_value = int(self.message_bits[:prefix_len], 2)
                cell_low = Decimal(prefix_value) / denominator
                cell_high = Decimal(prefix_value + 1) / denominator
                if self.low >= cell_low and self.high <= cell_high:
                    self._committed_bits = prefix_len
                else:
                    break
            return self._committed_bits

    @property
    def complete(self) -> bool:
        """Whether every message bit is guaranteed recoverable."""
        return self.bits_consumed == len(self.message_bits)

    def encode_step(self, dist: Sequence[float | Decimal]) -> int:
        """Given a probability distribution, return the token index to select."""
        with localcontext() as ctx:
            ctx.prec = self._precision
            probs = _normalize(dist)
            intervals = _partition_step(self.low, self.high, probs)

            for i, (lo, hi) in enumerate(intervals):
                if self.target >= lo and self.target < hi:
                    self.low = lo
                    self.high = hi
                    return i

            raise ArithmeticError("Target point fell outside the coding interval")


class ArithmeticDecoder:
    """Decodes token index selections back into the original message bits."""

    def __init__(self, num_bits: int) -> None:
        if num_bits < 0:
            raise ValueError("num_bits must be non-negative")
        self._precision = _precision_for_bits(num_bits)
        with localcontext() as ctx:
            ctx.prec = self._precision
            self.low = Decimal(0)
            self.high = Decimal(1)

    def decode_step(self, token_index: int, dist: Sequence[float | Decimal]) -> None:
        """Narrow the interval based on an observed token selection."""
        with localcontext() as ctx:
            ctx.prec = self._precision
            probs = _normalize(dist)
            if token_index < 0 or token_index >= len(probs):
                raise ValueError(
                    f"Token index {token_index} is outside the distribution"
                )
            intervals = _partition_step(self.low, self.high, probs)
            lo, hi = intervals[token_index]
            if lo >= hi:
                raise ValueError(f"Token index {token_index} has zero probability")
            self.low = lo
            self.high = hi

    def can_extract(self, num_bits: int) -> bool:
        """Return whether the interval lies inside one num_bits dyadic cell."""
        if num_bits < 0:
            raise ValueError("num_bits must be non-negative")
        if num_bits == 0:
            return True
        with localcontext() as ctx:
            ctx.prec = max(self._precision, _precision_for_bits(num_bits))
            denominator = Decimal(2) ** num_bits
            cell_index = int(self.low * denominator)
            cell_low = Decimal(cell_index) / denominator
            cell_high = Decimal(cell_index + 1) / denominator
            return self.low >= cell_low and self.high <= cell_high

    def extract_bits(self, num_bits: int) -> str:
        """Extract message bits from the narrowed interval.

        Uses the midpoint of [low, high) only after proving the whole interval
        lies within a single num_bits dyadic cell.
        """
        if not self.can_extract(num_bits):
            raise InsufficientCapacityError(
                f"Insufficient capacity: final interval does not guarantee {num_bits} bits"
            )
        with localcontext() as ctx:
            ctx.prec = max(self._precision, _precision_for_bits(num_bits))
            mid = (self.low + self.high) / 2
            return _point_to_bits(mid, num_bits)


def encode_message(
    secret_bits: str, distributions: Sequence[Sequence[float | Decimal]]
) -> list[int]:
    """Encode a bitstring into token selections using arithmetic coding.

    Args:
        secret_bits: The hidden message as a string of '0' and '1' characters.
        distributions: One probability distribution per generation step.

    Returns:
        List of token indices - one per distribution/step.

    Raises:
        ValueError: If distributions do not provide enough capacity for secret_bits.
    """
    prec = _precision_for_bits(len(secret_bits))
    with localcontext() as ctx:
        ctx.prec = prec

        encoder = ArithmeticEncoder(secret_bits)
        tokens: list[int] = []
        for dist in distributions:
            idx = encoder.encode_step(dist)
            tokens.append(idx)

        if not encoder.complete:
            raise InsufficientCapacityError(
                f"Insufficient capacity: distributions encoded only "
                f"{encoder.bits_consumed} bits but {len(secret_bits)} were required"
            )

    return tokens


def decode_message(
    token_indices: list[int],
    distributions: Sequence[Sequence[float | Decimal]],
    num_bits: int,
) -> str:
    """Decode token selections back into the hidden bitstring.

    Args:
        token_indices: The observed token indices (from encode_message).
        distributions: The same distributions used during encoding.
        num_bits: Number of message bits to extract.

    Returns:
        The recovered bitstring.

    Raises:
        ValueError: If token counts mismatch or interval width insufficient for num_bits.
    """
    if len(token_indices) != len(distributions):
        raise ValueError(
            f"Got {len(token_indices)} tokens but {len(distributions)} distributions"
        )

    prec = _precision_for_bits(num_bits)
    with localcontext() as ctx:
        ctx.prec = prec

        decoder = ArithmeticDecoder(num_bits)
        for idx, dist in zip(token_indices, distributions):
            decoder.decode_step(idx, dist)

        return decoder.extract_bits(num_bits)
