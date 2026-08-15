"""Tests for token stego fixes. Uses mock StegoModel - no GPU or real model needed."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.token_stego.activation_probe import (
    ActivationProbe,
    evaluate_probe,
    snapshot_to_features,
)
from src.token_stego.base import StegoOutput
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.detector import StegoDetector
from src.token_stego.latent import LatentSnapshot
from src.token_stego.model import StegoModel


def _make_mock_model(vocab_size=10):
    """Create a mock StegoModel with a small vocabulary."""
    model = MagicMock()
    model.vocab_size = vocab_size

    def mock_tokenize(text):
        if text.startswith("TOKENS:"):
            return [int(value) for value in text[7:].split(",") if value]
        # Return one token per character
        return list(range(min(len(text), vocab_size)))

    def mock_detokenize(ids):
        return "TOKENS:" + ",".join(str(value) for value in ids)

    def mock_get_distribution(
        input_ids, temperature=1.0, top_p=1.0, past_key_values=None, use_cache=False
    ):
        # Return a roughly uniform distribution over vocab
        dist = [1.0 / vocab_size] * vocab_size
        if use_cache:
            return dist, past_key_values
        return dist

    model.tokenize = mock_tokenize
    model.detokenize = mock_detokenize
    model.get_distribution = mock_get_distribution
    return model


class TestArithmeticStegoChannelBitsConsumed:
    """encode should return actual bits_consumed, not len(secret_bits)."""

    def test_bits_encoded_le_consumed(self):
        mock_model = _make_mock_model(vocab_size=4)
        channel = ArithmeticStegoChannel(model=mock_model)

        # Encode with max_tokens=1 - will likely not consume all bits
        result = channel.encode(secret="AB", prompt="test", max_tokens=1)

        # bits_encoded should be <= total secret bits (16 for "AB")
        total_secret_bits = 16  # 2 chars * 8 bits
        assert result.bits_encoded <= total_secret_bits
        assert isinstance(result, StegoOutput)
        assert result.tokens_generated == 1
        assert not result.complete

    def test_lossy_text_transport_never_returns_complete_output(self) -> None:
        mock_model = _make_mock_model(vocab_size=4)
        mock_model.detokenize = lambda _ids: "lossy"
        mock_model.tokenize = lambda _text: [0]
        channel = ArithmeticStegoChannel(model=mock_model)

        with pytest.raises(ValueError, match="text transport"):
            channel.encode(secret="A", prompt="test", max_tokens=8)


class TestArithmeticStegoChannelDecryptNonce:
    """decode should raise ValueError when key set but nonce missing."""

    def test_raises_without_nonce(self):
        mock_model = _make_mock_model(vocab_size=4)
        channel = ArithmeticStegoChannel(model=mock_model, key=b"testkey")

        with pytest.raises(ValueError, match="nonce"):
            channel.decode(text="AB", prompt="test", num_bits=16, nonce=None)


class TestStegoDetectorDeterminism:
    """score_cdf_uniformity should be deterministic with seed."""

    def test_seeded_determinism(self):
        mock_model = _make_mock_model(vocab_size=4)
        detector = StegoDetector(model=mock_model)

        r1 = detector.score_cdf_uniformity("AB", "test", seed=42)
        r2 = detector.score_cdf_uniformity("AB", "test", seed=42)

        assert r1.score == r2.score
        assert r1.p_value == r2.p_value
        assert r1.per_token_scores == r2.per_token_scores


class TestStegoModelDistributionContract:
    """Cached distribution calls always return a distribution/cache tuple."""

    def test_greedy_cached_call_returns_tuple(self) -> None:
        class FakeModel:
            device = torch.device("cpu")

            def __call__(self, *_args, **_kwargs):
                return SimpleNamespace(
                    logits=torch.tensor([[[0.1, 2.0, -1.0]]]),
                    past_key_values=("cached",),
                )

        model = StegoModel("fake")
        model._model = FakeModel()
        dist, cache = model.get_distribution([1], temperature=0, use_cache=True)

        assert dist == [0.0, 1.0, 0.0]
        assert cache == ("cached",)

    def test_invalid_sampling_parameters_fail_closed(self) -> None:
        class NeverCalledModel:
            device = torch.device("cpu")

            def __call__(self, *_args, **_kwargs):
                raise AssertionError(
                    "invalid arguments must fail before model execution"
                )

        model = StegoModel("fake")
        model._model = NeverCalledModel()

        for invalid_temperature in (-0.1, float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="temperature"):
                model.get_distribution([1], temperature=invalid_temperature)
        with pytest.raises(ValueError, match="top_p"):
            model.get_distribution([1], top_p=0.0)


class TestSnapshotToFeaturesMultiLayer:
    """snapshot_to_features should concatenate mean-pooled features from all layers."""

    def test_multi_layer_concatenation(self):
        snap = LatentSnapshot(
            hidden_states={
                0: torch.randn(1, 5, 3),
                6: torch.randn(1, 5, 3),
                12: torch.randn(1, 5, 3),
            },
        )
        features = snapshot_to_features(snap)
        # 3 layers * 3 (hidden_dim after mean over seq_len) = 9
        assert features.shape == (9,)

    def test_single_layer_extraction(self):
        snap = LatentSnapshot(
            hidden_states={
                0: torch.randn(1, 5, 3),
                6: torch.randn(1, 5, 3),
                12: torch.randn(1, 5, 3),
            },
        )
        features = snapshot_to_features(snap, layer=6)
        # Single layer: hidden_dim = 3
        assert features.shape == (3,)

    def test_single_layer_still_works(self):
        snap = LatentSnapshot(
            hidden_states={
                0: torch.randn(1, 5, 7),
            },
        )
        features = snapshot_to_features(snap, layer=0)
        assert features.shape == (7,)


class TestProbeUncertainty:
    def test_bootstrap_interval_is_seeded_and_reported(self) -> None:
        probe = ActivationProbe(input_dim=1, hidden_dim=1)
        with torch.no_grad():
            probe.net[0].weight.fill_(1.0)
            probe.net[0].bias.zero_()
            probe.net[2].weight.fill_(1.0)
            probe.net[2].bias.zero_()
        snapshots = [
            LatentSnapshot(hidden_states={0: torch.tensor([[[value]]])})
            for value in (-2.0, -1.0, 1.0, 2.0)
        ]
        labels = [0, 0, 1, 1]

        first = evaluate_probe(probe, snapshots, labels, seed=17)
        second = evaluate_probe(probe, snapshots, labels, seed=17)

        assert first["auc"] == 1.0
        assert first["auc_ci_95"] == [1.0, 1.0]
        assert first == second

    def test_auc_rejects_a_single_class_instead_of_reporting_chance(self) -> None:
        probe = ActivationProbe(input_dim=1, hidden_dim=1)
        snapshots = [
            LatentSnapshot(hidden_states={0: torch.tensor([[[value]]])})
            for value in (-2.0, -1.0)
        ]

        with pytest.raises(ValueError, match="both negative and positive"):
            evaluate_probe(probe, snapshots, [0, 0])

    def test_auc_rejects_boolean_labels(self) -> None:
        probe = ActivationProbe(input_dim=1, hidden_dim=1)
        snapshots = [
            LatentSnapshot(hidden_states={0: torch.tensor([[[value]]])})
            for value in (-2.0, -1.0, 1.0, 2.0)
        ]

        with pytest.raises(ValueError, match="binary values"):
            evaluate_probe(probe, snapshots, [False, 0, 1, 1])

    def test_group_bootstrap_requires_paired_classes(self) -> None:
        probe = ActivationProbe(input_dim=1, hidden_dim=1)
        snapshots = [
            LatentSnapshot(hidden_states={0: torch.tensor([[[value]]])})
            for value in (-2.0, -1.0, 1.0, 2.0)
        ]

        with pytest.raises(ValueError, match="both classes"):
            evaluate_probe(
                probe,
                snapshots,
                [0, 0, 1, 1],
                group_ids=["negative", "negative", "positive", "positive"],
            )
