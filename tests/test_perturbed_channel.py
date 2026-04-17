"""Tests for PerturbedStegoChannel round-trip recovery and save/load persistence."""

import tempfile
from unittest.mock import MagicMock

import torch

from src.token_stego.adversarial import (
    PerturbedStegoChannel,
    load_perturbation,
    save_perturbation,
)
from src.token_stego.channel import ArithmeticStegoChannel
from src.token_stego.perturbation import DistributionPerturbation

HIDDEN_DIM = 64
VOCAB_SIZE = 100


def _make_mock_model(
    hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE
) -> MagicMock:
    """Create a mock StegoModel with deterministic forward keyed on seq_len."""
    model = MagicMock()
    model.vocab_size = vocab_size

    inner = MagicMock()
    inner.device = torch.device("cpu")

    config = MagicMock()
    config.hidden_size = hidden_dim
    inner.config = config

    # Cache by seq_len so encoder and decoder see identical distributions
    # for the same context length.
    forward_cache: dict[int, MagicMock] = {}

    def forward_fn(input_ids: torch.Tensor, **_kwargs: object) -> MagicMock:
        seq_len = input_ids.shape[1]
        if seq_len not in forward_cache:
            gen = torch.Generator().manual_seed(seq_len * 7 + 3)
            result = MagicMock()
            result.logits = torch.randn(1, seq_len, vocab_size, generator=gen)
            result.hidden_states = [
                torch.randn(1, seq_len, hidden_dim, generator=gen) for _ in range(3)
            ]
            forward_cache[seq_len] = result
        return forward_cache[seq_len]

    inner.side_effect = forward_fn
    model.model = inner

    model.tokenize = MagicMock(return_value=[1, 2, 3])
    model.detokenize = MagicMock(return_value="mock text")

    # Mock get_distribution to support KV-cache protocol
    dist_cache: dict[int, list[float]] = {}

    def mock_get_distribution(
        input_ids, temperature=1.0, top_p=1.0, past_key_values=None, use_cache=False
    ):
        seq_len = len(input_ids)
        if seq_len not in dist_cache:
            gen = torch.Generator().manual_seed(seq_len * 7 + 3)
            logits = torch.randn(vocab_size, generator=gen)
            probs = torch.softmax(logits, dim=0)
            dist_cache[seq_len] = probs.tolist()
        dist = dist_cache[seq_len]
        if use_cache:
            return dist, past_key_values
        return dist

    model.get_distribution = mock_get_distribution

    return model


# --- Round-trip recovery tests (C1, C2) ---


class TestRoundTripRecovery:
    """Verify encode then decode produces identical bits."""

    def test_exact_recovery_single_char(self) -> None:
        """C1: encode/decode round-trip achieves BER=0.0 for a single ASCII char."""
        torch.manual_seed(42)
        mock_model = _make_mock_model()
        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        channel = PerturbedStegoChannel(inner, perturbation)

        secret = "A"
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        encoded = channel.encode(secret, "prompt", max_tokens=50)

        # Reset tokenize mock for decode: first call returns stego tokens,
        # second returns prompt tokens.
        mock_model.tokenize = MagicMock(side_effect=[encoded.tokens, [1, 2, 3]])

        recovered = channel.decode(encoded.text, "prompt", encoded.bits_encoded)
        assert recovered == secret_bits, (
            f"BER != 0.0: expected {secret_bits}, got {recovered}"
        )

    def test_exact_recovery_multi_char(self) -> None:
        """C1: round-trip works for a multi-character secret."""
        torch.manual_seed(42)
        mock_model = _make_mock_model()
        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        channel = PerturbedStegoChannel(inner, perturbation)

        secret = "OK"
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        encoded = channel.encode(secret, "prompt", max_tokens=100)

        mock_model.tokenize = MagicMock(side_effect=[encoded.tokens, [1, 2, 3]])

        recovered = channel.decode(encoded.text, "prompt", encoded.bits_encoded)
        assert recovered == secret_bits

    def test_exact_recovery_with_encryption(self) -> None:
        """C1: round-trip works with encryption enabled."""
        torch.manual_seed(42)
        mock_model = _make_mock_model()
        key = b"test-key-1234567"
        inner = ArithmeticStegoChannel(model=mock_model, key=key)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        channel = PerturbedStegoChannel(inner, perturbation)

        secret = "Z"
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        encoded = channel.encode(secret, "prompt", max_tokens=50)

        mock_model.tokenize = MagicMock(side_effect=[encoded.tokens, [1, 2, 3]])

        recovered = channel.decode(
            encoded.text, "prompt", encoded.bits_encoded, nonce=encoded.nonce
        )
        assert recovered == secret_bits

    def test_ber_is_zero(self) -> None:
        """C1: explicitly compute BER and assert it is 0.0."""
        torch.manual_seed(42)
        mock_model = _make_mock_model()
        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        channel = PerturbedStegoChannel(inner, perturbation)

        secret = "B"
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        encoded = channel.encode(secret, "prompt", max_tokens=50)

        mock_model.tokenize = MagicMock(side_effect=[encoded.tokens, [1, 2, 3]])

        recovered = channel.decode(encoded.text, "prompt", encoded.bits_encoded)
        errors = sum(a != b for a, b in zip(secret_bits, recovered))
        ber = errors / len(secret_bits) if secret_bits else 1.0
        assert ber == 0.0, f"BER={ber}, expected 0.0"


class TestDistributionReplay:
    """C2: Decoder must replay identical perturbed distributions as encoder."""

    def test_distributions_identical_between_encode_and_decode(self) -> None:
        """Capture distributions during encode and decode, verify they match."""
        torch.manual_seed(42)
        mock_model = _make_mock_model()
        inner = ArithmeticStegoChannel(model=mock_model)
        perturbation = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        channel = PerturbedStegoChannel(inner, perturbation)

        # Capture distributions during encode
        encode_dists: list[list[float]] = []
        original_get_dist = channel._get_perturbed_distribution

        def capture_encode(context_ids: list[int]) -> list[float]:
            dist = original_get_dist(context_ids)
            encode_dists.append(dist)
            return dist

        channel._get_perturbed_distribution = capture_encode  # type: ignore[assignment]

        secret = "X"
        encoded = channel.encode(secret, "prompt", max_tokens=50)

        # Reset for decode
        channel._get_perturbed_distribution = original_get_dist  # type: ignore[assignment]

        # Capture distributions during decode
        decode_dists: list[list[float]] = []

        def capture_decode(context_ids: list[int]) -> list[float]:
            dist = original_get_dist(context_ids)
            decode_dists.append(dist)
            return dist

        channel._get_perturbed_distribution = capture_decode  # type: ignore[assignment]

        mock_model.tokenize = MagicMock(side_effect=[encoded.tokens, [1, 2, 3]])
        channel.decode(encoded.text, "prompt", encoded.bits_encoded)

        assert len(encode_dists) == len(decode_dists), (
            f"Distribution count mismatch: encode={len(encode_dists)}, "
            f"decode={len(decode_dists)}"
        )

        for i, (ed, dd) in enumerate(zip(encode_dists, decode_dists)):
            assert ed == dd, f"Distribution mismatch at step {i}"


# --- Save/load tests (C3, C4) ---


class TestSaveLoadPerturbation:
    """Tests for save_perturbation and load_perturbation."""

    def test_save_and_load_are_callable(self) -> None:
        """C3: Both functions exist and are callable."""
        assert callable(save_perturbation)
        assert callable(load_perturbation)

    def test_save_creates_file(self) -> None:
        """save_perturbation writes a file to disk."""
        net = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        save_perturbation(net, path)
        import os

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        os.unlink(path)

    def test_load_returns_distribution_perturbation(self) -> None:
        """load_perturbation returns a DistributionPerturbation instance."""
        net = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        save_perturbation(net, path)
        loaded = load_perturbation(path)
        assert isinstance(loaded, DistributionPerturbation)
        import os

        os.unlink(path)

    def test_weights_bitwise_equal_after_roundtrip(self) -> None:
        """C4: Save then load produces identical model weights."""
        torch.manual_seed(123)
        net = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE, bottleneck=32)
        # Mutate weights away from init to make the test meaningful
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        save_perturbation(net, path)
        loaded = load_perturbation(path)

        for (name, orig_param), (_, loaded_param) in zip(
            net.named_parameters(), loaded.named_parameters()
        ):
            assert torch.equal(orig_param, loaded_param), (
                f"Parameter {name} differs after save/load"
            )

        import os

        os.unlink(path)

    def test_custom_bottleneck_preserved(self) -> None:
        """Architecture metadata (bottleneck) survives save/load."""
        net = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE, bottleneck=16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        save_perturbation(net, path)
        loaded = load_perturbation(path)

        # Verify the bottleneck dimension by checking the first layer's out_features
        first_layer = loaded.net[0]
        assert isinstance(first_layer, torch.nn.Linear)
        assert first_layer.out_features == 16

        import os

        os.unlink(path)

    def test_loaded_model_in_eval_mode(self) -> None:
        """Loaded perturbation should be in eval mode."""
        net = DistributionPerturbation(HIDDEN_DIM, VOCAB_SIZE)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        save_perturbation(net, path)
        loaded = load_perturbation(path)
        assert not loaded.training

        import os

        os.unlink(path)
