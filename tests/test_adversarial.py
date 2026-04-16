"""Tests for adversarial training loop and PerturbedStegoChannel."""

from unittest.mock import MagicMock

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

    return model


def _make_probe(hidden_dim: int = HIDDEN_DIM) -> ActivationProbe:
    """Create a probe with random weights."""
    probe = ActivationProbe(input_dim=hidden_dim, hidden_dim=32)
    probe.eval()
    return probe


# --- train_perturbation tests ---


class TestTrainPerturbation:
    """Tests for the adversarial training loop."""

    def test_returns_trained_perturbation(self) -> None:
        """train_perturbation returns a DistributionPerturbation instance."""
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3], [4, 5, 6]]
        config = AdversarialTrainingConfig(steps=5, lr=1e-3)

        result = train_perturbation(model, probe, contexts, config)

        assert isinstance(result.perturbation, DistributionPerturbation)

    def test_loss_decreases(self) -> None:
        """C1: Final loss should be lower than initial loss after training."""
        torch.manual_seed(42)
        model = _make_mock_model()
        probe = _make_probe()
        # Use a single context so the model returns consistent shapes
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=50, lr=1e-2, kl_weight=0.01)

        result = train_perturbation(model, probe, contexts, config)

        # Compare early vs late average (more robust than single-point comparison)
        early_avg = sum(result.loss_history[:5]) / 5
        late_avg = sum(result.loss_history[-5:]) / 5
        assert late_avg < early_avg, (
            f"Loss did not decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )

    def test_kl_divergence_non_negative(self) -> None:
        """C2: KL divergence should be non-negative for all training steps."""
        torch.manual_seed(42)
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3], [4, 5, 6]]
        config = AdversarialTrainingConfig(steps=20, lr=1e-3)

        result = train_perturbation(model, probe, contexts, config)

        for step_idx, kl_val in enumerate(result.kl_history):
            assert kl_val >= -1e-6, f"Negative KL at step {step_idx}: {kl_val}"

    def test_model_params_frozen(self) -> None:
        """C3: LLM parameters should not receive gradients during training."""
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=3)

        # The mock model's forward is called inside torch.no_grad(),
        # and outputs are detached. Verify by checking the call pattern.
        train_perturbation(model, probe, contexts, config)

        # The model forward is called via _forward_with_hidden which uses
        # torch.no_grad(). We verify by checking that the model's __call__
        # was invoked (it was) and that outputs are detached in the implementation.
        # Direct param check: mock doesn't have real params, so we verify
        # the code structure instead via a separate focused test below.

    def test_probe_params_frozen_during_training(self) -> None:
        """C4: Probe parameters should have requires_grad=False during training."""
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=3)

        # Track probe param grad state during training
        grad_states_during: list[bool] = []
        original_forward = probe.forward

        def tracking_forward(x: torch.Tensor) -> torch.Tensor:
            for p in probe.parameters():
                grad_states_during.append(p.requires_grad)
            return original_forward(x)

        probe.forward = tracking_forward  # type: ignore[assignment]
        train_perturbation(model, probe, contexts, config)

        # All recorded states should be False
        assert len(grad_states_during) > 0, "Probe forward was never called"
        assert all(not g for g in grad_states_during), (
            f"Probe had requires_grad=True during training: {grad_states_during}"
        )

    def test_probe_grad_restored_after_training(self) -> None:
        """Probe parameters should have requires_grad restored after training."""
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=3)

        train_perturbation(model, probe, contexts, config)

        for p in probe.parameters():
            assert p.requires_grad, "Probe requires_grad not restored after training"

    def test_only_perturbation_params_updated(self) -> None:
        """Only the perturbation network's parameters should change."""
        torch.manual_seed(42)
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=10, lr=1e-2)

        # Snapshot probe params before training
        probe_before = {name: p.clone() for name, p in probe.named_parameters()}

        result = train_perturbation(model, probe, contexts, config)

        # Probe params should be unchanged
        for name, p in probe.named_parameters():
            assert torch.equal(p.data, probe_before[name]), (
                f"Probe parameter {name} was modified during training"
            )

        # Perturbation params should have changed from their initialization
        # (non-trivial because we ran training steps with non-zero LR)
        net = result.perturbation
        all_zero = all(torch.all(p == 0).item() for p in net.parameters())
        assert not all_zero, "Perturbation net params are all zero after training"

    def test_loss_history_length_matches_steps(self) -> None:
        """Loss history should have one entry per training step."""
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=15)

        result = train_perturbation(model, probe, contexts, config)

        assert len(result.loss_history) == 15
        assert len(result.kl_history) == 15
        assert len(result.adversarial_history) == 15


class TestModelFreezing:
    """Verify that LLM forward pass happens under no_grad."""

    def test_forward_with_hidden_uses_no_grad_context(self) -> None:
        """C3: _forward_with_hidden is called inside no_grad in train_perturbation."""
        model = _make_mock_model()
        probe = _make_probe()
        contexts = [[1, 2, 3]]
        config = AdversarialTrainingConfig(steps=2)

        grad_enabled_during_forward: list[bool] = []
        original_call = model.model.side_effect

        def tracking_call(input_ids: torch.Tensor, **kwargs: object) -> MagicMock:
            grad_enabled_during_forward.append(torch.is_grad_enabled())
            return original_call(input_ids, **kwargs)

        model.model.side_effect = tracking_call
        train_perturbation(model, probe, contexts, config)

        assert len(grad_enabled_during_forward) > 0
        assert all(not g for g in grad_enabled_during_forward), (
            "Model forward was called with grad enabled"
        )


# --- PerturbedStegoChannel tests ---


class TestPerturbedStegoChannel:
    """Tests for PerturbedStegoChannel wrapping ArithmeticStegoChannel."""

    def test_has_encode_and_decode(self) -> None:
        """C5: PerturbedStegoChannel must have encode and decode methods."""
        assert hasattr(PerturbedStegoChannel, "encode")
        assert hasattr(PerturbedStegoChannel, "decode")
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
        # Mock tokenize/detokenize for the channel
        mock_model.tokenize = MagicMock(return_value=[1, 2, 3])
        mock_model.detokenize = MagicMock(return_value="generated text")

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
        mock_model.tokenize = MagicMock(return_value=[1, 2, 3])
        mock_model.detokenize = MagicMock(return_value="fake text")

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
        mock_model.tokenize = MagicMock(return_value=[1, 2, 3])
        mock_model.detokenize = MagicMock(return_value="text")

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
