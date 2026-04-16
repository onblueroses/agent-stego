"""Tests for distribution perturbation network."""

import torch

from src.token_stego.perturbation import DistributionPerturbation, apply_perturbation


class TestDistributionPerturbation:
    """Test the perturbation network architecture."""

    def test_forward_shape(self) -> None:
        """Output shape matches vocab_size."""
        hidden_dim = 128
        vocab_size = 1000
        net = DistributionPerturbation(hidden_dim, vocab_size)
        x = torch.randn(hidden_dim)
        out = net(x)
        assert out.shape == (vocab_size,)

    def test_forward_batched(self) -> None:
        """Works with batched input."""
        hidden_dim = 128
        vocab_size = 1000
        net = DistributionPerturbation(hidden_dim, vocab_size)
        x = torch.randn(4, hidden_dim)
        out = net(x)
        assert out.shape == (4, vocab_size)

    def test_zero_init_identity(self) -> None:
        """Zero-initialized perturbation produces unchanged distribution."""
        hidden_dim = 64
        vocab_size = 100
        net = DistributionPerturbation(hidden_dim, vocab_size)
        # Zero out all weights
        for p in net.parameters():
            p.data.zero_()

        logits = torch.randn(vocab_size)
        hidden = torch.randn(hidden_dim)

        perturbed = apply_perturbation(logits, hidden, net)
        original = torch.softmax(logits, dim=0)

        assert torch.allclose(perturbed, original, atol=1e-5)

    def test_valid_distribution(self) -> None:
        """Perturbed output sums to 1 and is non-negative."""
        hidden_dim = 64
        vocab_size = 100
        net = DistributionPerturbation(hidden_dim, vocab_size)
        logits = torch.randn(vocab_size)
        hidden = torch.randn(hidden_dim)

        perturbed = apply_perturbation(logits, hidden, net)

        assert perturbed.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert (perturbed >= 0).all()

    def test_kl_bounded_small_weights(self) -> None:
        """KL divergence is small when perturbation weights are small."""
        hidden_dim = 64
        vocab_size = 100
        net = DistributionPerturbation(hidden_dim, vocab_size)
        # Scale weights to be tiny
        for p in net.parameters():
            p.data.mul_(0.001)

        logits = torch.randn(vocab_size)
        hidden = torch.randn(hidden_dim)

        original = torch.softmax(logits, dim=0)
        perturbed = apply_perturbation(logits, hidden, net)

        # KL(original || perturbed)
        kl = torch.sum(original * (original.log() - perturbed.clamp(min=1e-10).log()))
        assert kl.item() < 0.1  # small perturbation -> small KL

    def test_perturbation_changes_distribution(self) -> None:
        """Non-trivial perturbation modifies the distribution."""
        hidden_dim = 64
        vocab_size = 100
        net = DistributionPerturbation(hidden_dim, vocab_size)
        # Scale up weights so the effect is clearly visible
        for p in net.parameters():
            p.data.mul_(100.0)

        logits = torch.randn(vocab_size)
        hidden = torch.randn(hidden_dim)

        original = torch.softmax(logits, dim=0)
        perturbed = apply_perturbation(logits, hidden, net)

        assert not torch.allclose(perturbed, original, atol=1e-3)

    def test_gradient_flows_through_perturbation(self) -> None:
        """Gradients flow through apply_perturbation to the network params."""
        hidden_dim = 64
        vocab_size = 100
        net = DistributionPerturbation(hidden_dim, vocab_size)
        logits = torch.randn(vocab_size)
        hidden = torch.randn(hidden_dim)

        perturbed = apply_perturbation(logits, hidden, net)
        # Use a non-trivial loss (not sum, which is constant=1 for softmax)
        loss = (perturbed * torch.randn(vocab_size)).sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )
        assert has_grad


# Need pytest for approx
import pytest  # noqa: E402
