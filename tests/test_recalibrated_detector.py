"""Tests for the recalibrated output-level detector (mock distributions)."""

import math

from src.token_stego.recalibrated_detector import (
    RecalibratedDetector,
    mean_normalized_surprise,
)


class _FakeModel:
    """Deterministic distribution: token 0 is high-probability, the rest low."""

    def __init__(self, vocab: int = 8) -> None:
        self.vocab = vocab
        self._probs = [0.6] + [0.4 / (vocab - 1)] * (vocab - 1)

    def tokenize(self, text: str) -> list[int]:
        return [int(c) for c in text if c.isdigit()]

    def get_distribution(
        self,
        input_ids: list[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> list[float]:
        return list(self._probs)


class TestMeanNormalizedSurprise:
    def test_high_probability_picks_below_one(self) -> None:
        """Always picking the argmax token yields surprise well below 1."""
        model = _FakeModel()
        surprise = mean_normalized_surprise(model, "000000", "12")
        entropy = -sum(p * math.log(p + 1e-30) for p in model._probs if p > 0)
        expected = (-math.log(0.6)) / (entropy + 1e-10)
        assert surprise == expected
        assert surprise < 1.0

    def test_spread_picks_higher_surprise(self) -> None:
        """Picking tokens across the distribution raises mean surprise."""
        model = _FakeModel()
        high_prob = mean_normalized_surprise(model, "000000", "12")
        spread = mean_normalized_surprise(model, "01234567", "12")
        assert spread > high_prob

    def test_empty_text_returns_one(self) -> None:
        model = _FakeModel()
        assert mean_normalized_surprise(model, "", "12") == 1.0


class TestRecalibratedDetector:
    def test_flag_low_surprise(self) -> None:
        detector = RecalibratedDetector(threshold=0.95)
        assert detector.flag(0.8) is True
        assert detector.flag(1.0) is False

    def test_score_monotonic(self) -> None:
        detector = RecalibratedDetector(threshold=1.0)
        assert detector.score(0.5) == 0.5
        assert detector.score(1.0) == 0.0
        assert detector.score(1.2) == 0.0

    def test_detect_round_trip(self) -> None:
        model = _FakeModel()
        detector = RecalibratedDetector(threshold=0.9)
        flagged, surprise = detector.detect(model, "000000", "12")
        assert flagged is True
        assert surprise < 0.9
