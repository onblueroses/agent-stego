"""Tests for adversarial training loop and PerturbedStegoChannel."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch

from src.token_stego.activation_probe import ActivationProbe
from src.token_stego.adversarial import (
    AdversarialTrainingConfig,
    PerturbedStegoChannel,
    train_perturbation,
)
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.perturbation import DistributionPerturbation

# --- Fixtures ---


HIDDEN_DIM = 64
VOCAB_SIZE = 100


def _make_mock_model(
    hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE
) -> MagicMock:
    """Create a mock StegoModel that returns synthetic logits and hidden states."""
    model = MagicMock()
    model.vocab_size = vocab_size

    # Mock the inner HF model
    inner = MagicMock()
    inner.device = torch.device("cpu")

    config = MagicMock()
    config.hidden_size = hidden_dim
    inner.config = config

    # Cache outputs by input shape so repeated calls with the same context
    # return identical tensors - required for training convergence.
    forward_cache: dict[tuple[int, int], MagicMock] = {}

    def forward_fn(input_ids: torch.Tensor, **kwargs: object) -> MagicMock:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        cache_key = (batch_size, seq_len)
        if cache_key not in forward_cache:
            result = MagicMock()
            result.logits = torch.randn(batch_size, seq_len, vocab_size)
            result.hidden_states = [
                torch.randn(batch_size, seq_len, hidden_dim) for _ in range(3)
            ]
            forward_cache[cache_key] = result
        return forward_cache[cache_key]

    inner.side_effect = forward_fn
    model.model = inner

    def mock_tokenize(text: str) -> list[int]:
        if text.startswith("TOKENS:"):
            return [int(value) for value in text[7:].split(",") if value]
        return [1, 2, 3]

    model.tokenize = MagicMock(side_effect=mock_tokenize)
    model.detokenize = MagicMock(
        side_effect=lambda ids: "TOKENS:" + ",".join(str(value) for value in ids)
    )

    return model


def _make_probe(hidden_dim: int = HIDDEN_DIM) -> ActivationProbe:
    """Create a probe with random weights."""
    probe = ActivationProbe(input_dim=hidden_dim, hidden_dim=32)
    probe.eval()
    return probe


# --- train_perturbation tests ---


class TestTrainPerturbation:
    """The causally invalid historical trainer must fail closed."""

    def test_retired_surrogate_raises_before_model_execution(self) -> None:
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=2)

        with pytest.raises(RuntimeError, match="causal hidden-state"):
            train_perturbation(model, probe, contexts, config)
        model.model.assert_not_called()

    @pytest.mark.parametrize(
        "module_name",
        ["experiments.adversarial_probe", "experiments.adversarial_sweep"],
    )
    def test_retired_experiment_modules_import_but_cannot_run(
        self, module_name: str
    ) -> None:
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        with pytest.raises(SystemExit, match="causal hidden-state"):
            module.main()


# --- PerturbedStegoChannel tests ---


class TestPerturbedStegoChannel:
    """Tests for PerturbedStegoChannel wrapping ArithmeticStegoChannel."""

    def test_has_encode_and_decode(self) -> None:
        """C5: PerturbedStegoChannel must have encode and decode methods."""
        assert hasattr(PerturbedStegoChannel, "encode")
        assert hasattr(PerturbedStegoChannel, "decode")

    @pytest.mark.parametrize(
        ("temperature", "top_p"),
        [
            (float("nan"), 1.0),
            (float("inf"), 1.0),
            (float("-inf"), 1.0),
            (-0.1, 1.0),
            (1.0, 0.0),
            (1.0, 1.1),
            (1.0, float("nan")),
            (1.0, float("inf")),
        ],
    )
    def test_invalid_sampling_fails_before_model_access(
        self, temperature: float, top_p: float
    ) -> None:
        inner = MagicMock()
        inner._temperature = 1.0
        inner._top_p = 1.0
        inner._model = MagicMock()
        perturbation = MagicMock()

        with pytest.raises(ValueError):
            PerturbedStegoChannel(
                inner,
                perturbation,
                temperature=temperature,
                top_p=top_p,
            )
        assert not inner._model.mock_calls
        perturbation.to.assert_not_called()
        assert callable(PerturbedStegoChannel.encode)
        assert callable(PerturbedStegoChannel.decode)

    def test_wraps_arithmetic_channel(self) -> None:
        """PerturbedStegoChannel wraps an ArithmeticStegoChannel."""
        mock_model = _make_mock_model()
        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)

        channel = PerturbedStegoChannel(inner, perturbation)

        assert channel._inner is inner
        assert channel._perturbation is perturbation

    def test_encode_returns_stego_output(self) -> None:
        """encode() returns a StegoOutput with expected fields."""
        from src.token_stego.base import StegoOutput

        mock_model = _make_mock_model()

        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)

        channel = PerturbedStegoChannel(inner, perturbation)
        result = channel.encode("hi", "test prompt", max_tokens=5)

        assert isinstance(result, StegoOutput)
        assert result.tokens_generated > 0
        assert result.bits_encoded > 0

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding then decoding with same perturbation recovers bits."""
        torch.manual_seed(99)
        mock_model = _make_mock_model()

        # We need deterministic forward passes for encode/decode to agree.
        # Use a fixed set of outputs keyed by context length.
        forward_cache: dict[int, MagicMock] = {}

        def deterministic_forward(
            input_ids: torch.Tensor, **kwargs: object
        ) -> MagicMock:
            seq_len = input_ids.shape[1]
            if seq_len not in forward_cache:
                result = MagicMock()
                result.logits = torch.randn(1, seq_len, VOCAB_SIZE)
                result.hidden_states = [
                    torch.randn(1, seq_len, HIDDEN_DIM) for _ in range(3)
                ]
                forward_cache[seq_len] = result
            return forward_cache[seq_len]

        mock_model.model.side_effect = deterministic_forward

        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)

        channel = PerturbedStegoChannel(inner, perturbation)

        secret = "A"  # 8 bits
        encoded = channel.encode(secret, "prompt", max_tokens=50)

        # For decode, tokenize must return the encoded tokens
        mock_model.tokenize = MagicMock(
            side_effect=[
                encoded.tokens,  # tokenize(text) for stego_ids
                [1, 2, 3],  # tokenize(prompt) for context_ids
            ]
        )

        recovered_bits = channel.decode(encoded.text, "prompt", encoded.bits_encoded)
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        assert recovered_bits == secret_bits

    def test_perturbation_applied_during_encode(self) -> None:
        """Verify that the perturbation network is actually called during encode."""
        mock_model = _make_mock_model()

        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)

        # Track calls to perturbation forward
        call_count = 0
        original_forward = perturbation.forward

        def counting_forward(x: torch.Tensor) -> torch.Tensor:
            nonlocal call_count
            call_count += 1
            return original_forward(x)

        perturbation.forward = counting_forward  # type: ignore[assignment]

        channel = PerturbedStegoChannel(inner, perturbation)
        channel.encode("A", "prompt", max_tokens=5)

        assert call_count > 0, "Perturbation network was never called during encode"

    def test_lossy_text_transport_never_returns_complete_output(self) -> None:
        mock_model = _make_mock_model()
        mock_model.detokenize = MagicMock(return_value="lossy")
        mock_model.tokenize = MagicMock(return_value=[1, 2, 3])
        inner = ArithmeticStegoChannel(model=mock_model)
        channel = PerturbedStegoChannel(
            inner, DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        )

        with pytest.raises(ValueError, match="text transport"):
            channel.encode("A", "prompt", max_tokens=50)
