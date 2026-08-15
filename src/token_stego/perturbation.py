"""Distribution perturbation for adversarial probe evasion.

Learns a small network that adjusts the LLM's logits before they enter
the arithmetic encoder. The goal: make stego activations look like normal
activations to a trained probe, while keeping the perturbed distribution
close enough to the original that message recovery still works.
"""

import math

import torch
from torch import Tensor, nn

# Retained from the historical perturbation architecture for replay only. The
# trainer that treated these values as effective has been retired.
DEFAULT_BOTTLENECK = 64
INITIAL_OUTPUT_SCALE = 0.01


class DistributionPerturbation(nn.Module):
    """Small MLP that maps hidden states to logit adjustments.

    Applied additively in logit space before softmax, so the output
    is always a valid probability distribution.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        bottleneck: int = DEFAULT_BOTTLENECK,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, vocab_size),
        )
        # Initialize near-zero so the perturbation starts as identity
        output_layer = self.net[-1]
        assert isinstance(output_layer, nn.Linear)
        with torch.no_grad():
            output_layer.weight.mul_(INITIAL_OUTPUT_SCALE)
            output_layer.bias.mul_(INITIAL_OUTPUT_SCALE)

    def forward(self, hidden_state: Tensor) -> Tensor:
        """Map hidden state to logit adjustments.

        Args:
            hidden_state: (hidden_dim,) or (batch, hidden_dim)

        Returns:
            Logit adjustments, same trailing dim as vocab_size.
        """
        return self.net(hidden_state)


def apply_perturbation(
    logits: Tensor,
    hidden_state: Tensor,
    perturbation: DistributionPerturbation,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """Apply perturbation to logits and return a probability distribution.

    Args:
        logits: Raw model logits, shape (vocab_size,) or (batch, vocab_size).
        hidden_state: Hidden state vector for the perturbation net.
        perturbation: The perturbation network.
        temperature: Softmax temperature (applied after perturbation).
        top_p: Nucleus filtering threshold.

    Returns:
        Probability distribution tensor, same shape as logits.
    """
    validate_sampling_parameters(temperature, top_p)
    adjustment = perturbation(hidden_state)
    adjusted_logits = logits + adjustment

    if temperature == 0:
        probs = torch.zeros_like(adjusted_logits)
        probs.scatter_(-1, adjusted_logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    if temperature != 1.0:
        adjusted_logits = adjusted_logits / temperature

    probs = torch.softmax(adjusted_logits, dim=-1)

    if 0 < top_p < 1.0:
        probs = _apply_top_p(probs, top_p)

    return probs


def validate_sampling_parameters(temperature: float, top_p: float) -> None:
    """Reject non-finite or out-of-range sampling controls."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1.0:
        raise ValueError("top_p must be finite and in the interval (0, 1]")


def _apply_top_p(probs: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) filtering and renormalize."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = (cumsum - sorted_probs) >= top_p
    sorted_probs = torch.where(
        cutoff_mask, torch.zeros_like(sorted_probs), sorted_probs
    )

    # Scatter back
    result = torch.zeros_like(probs)
    result.scatter_(-1, sorted_indices, sorted_probs)
    total = result.sum(dim=-1, keepdim=True)
    result = result / total.clamp(min=1e-10)
    return result
