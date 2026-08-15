"""Recalibrated output-level detector for the token-likelihood coder.

The original StegoDetector (detector.py) computes
`abs(mean(normalized_surprise) - 1)` and is inverted on real data
(rounds 31/37): it flags normal text as stego. The underlying signal is
directional — the arithmetic coder selects high-probability tokens, so stego
text has mean normalized surprise *below* normal sampling. This module
exposes the raw feature and a direction-aware detector:

- feature: mean over tokens of (-log p(token) / entropy)
- measured (2026-08-14, Qwen2.5-0.5B): stego = 0.99 (the native-sampling
  expectation), T=0.8 normal = 0.45, greedy = 0.29 — the coder is
  surprise-preserving at matched temperature, so this feature cannot
  separate it from matched sampling (see analysis/recalibrated_detector.md).

The module is the reusable, offline-testable implementation with
mock-distribution tests; the measured finding means a surprise-based output
detector has no signal against this coder, by design.

CDF-uniformity is excluded by design: a pseudorandom coder target makes PIT
draws uniform, so that test cannot separate this coder from sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class TokenDistributionSource(Protocol):
    """Anything that can tokenize text and return next-token distributions.

    StegoModel satisfies this structurally; mocks do too, which keeps the
    detector testable offline.
    """

    def tokenize(self, text: str) -> list[int]: ...

    def get_distribution(
        self,
        input_ids: list[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: object,
    ) -> list[float]: ...


def mean_normalized_surprise(
    model: TokenDistributionSource, text: str, prompt: str
) -> float:
    """Mean of -log p(token) / entropy over the generated tokens.

    ~1.0 under normal sampling; below 1.0 when tokens are systematically
    higher-probability than a sample (the coder's selection behavior).
    """
    token_ids = model.tokenize(text)
    context = list(model.tokenize(prompt))
    values: list[float] = []
    for token_id in token_ids:
        dist = model.get_distribution(context)
        entropy = -sum(p * math.log(p + 1e-30) for p in dist if p > 0)
        prob = dist[token_id]
        if prob <= 0:
            values.append(1.0)
        else:
            values.append((-math.log(prob)) / (entropy + 1e-10))
        context.append(token_id)
    if not values:
        return 1.0
    return sum(values) / len(values)


@dataclass(frozen=True)
class RecalibratedDetector:
    """Direction-aware entropy detector: low mean surprise flags stego.

    threshold is the boundary below which a text is flagged. Direction:
    stego surprise < normal surprise, so greater_is_suspicious=False.
    """

    threshold: float = 1.0

    def score(self, surprise: float) -> float:
        """Anomaly score in [0, 1]: 1 when surprise is far below threshold."""
        if surprise >= self.threshold:
            return 0.0
        return min(1.0, (self.threshold - surprise) / self.threshold)

    def flag(self, surprise: float) -> bool:
        return surprise < self.threshold

    def detect(
        self, model: TokenDistributionSource, text: str, prompt: str
    ) -> tuple[bool, float]:
        surprise = mean_normalized_surprise(model, text, prompt)
        return self.flag(surprise), surprise
