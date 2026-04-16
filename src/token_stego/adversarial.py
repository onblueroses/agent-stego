"""Adversarial training loop for token-likelihood steganography evasion.

Trains a DistributionPerturbation network to fool a frozen ActivationProbe
into misclassifying stego inference as normal. The perturbation is applied
in logit space (additive before softmax) so the result is always a valid
probability distribution.

Also provides PerturbedStegoChannel, which wraps ArithmeticStegoChannel
with a trained perturbation network for evasion during encode/decode.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .activation_probe import ActivationProbe
from .base import StegoOutput
from .channel import ArithmeticStegoChannel
from .coding import decode_message
from .crypto import decrypt_bits, encrypt_bits
from .model import StegoModel
from .perturbation import DistributionPerturbation, apply_perturbation


@dataclass
class AdversarialTrainingConfig:
    """Configuration for adversarial perturbation training."""

    steps: int = 200
    lr: float = 1e-3
    kl_weight: float = 0.1
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass
class AdversarialTrainingResult:
    """Training outcome with loss history for diagnostics."""

    perturbation: DistributionPerturbation
    loss_history: list[float]
    kl_history: list[float]
    adversarial_history: list[float]


def train_perturbation(
    model: StegoModel,
    probe: ActivationProbe,
    train_contexts: list[list[int]],
    config: AdversarialTrainingConfig | None = None,
) -> AdversarialTrainingResult:
    """Train a perturbation network to evade a frozen activation probe.

    The training loop:
    1. Forward pass through frozen LLM to get logits + hidden states
    2. Apply perturbation network (with grad) to get perturbed distribution
    3. Forward through frozen probe to get detection score
    4. Composite loss = adversarial(probe_score) + kl_weight * KL(original || perturbed)
    5. Backprop through perturbation net only

    Args:
        model: Loaded StegoModel. Parameters are frozen (no_grad) during training.
        probe: Trained ActivationProbe. Parameters are frozen during training.
        train_contexts: List of token ID sequences to use as training contexts.
            Each sequence is fed to the LLM to produce logits and hidden states.
        config: Training hyperparameters.

    Returns:
        AdversarialTrainingResult with trained perturbation and loss history.
    """
    if config is None:
        config = AdversarialTrainingConfig()

    hidden_dim = _get_hidden_dim(model)
    vocab_size = model.vocab_size

    # Derive device from actual model outputs, not model.device attribute.
    # With device_map="auto", model.device points at the first shard but
    # activations come from the last shard.
    with torch.no_grad():
        probe_ids = train_contexts[0][:1] or [0]
        probe_logits, _ = _forward_with_hidden(model, probe_ids)
        device = probe_logits.device

    perturbation_net = DistributionPerturbation(hidden_dim, vocab_size).to(device)
    optimizer = torch.optim.Adam(perturbation_net.parameters(), lr=config.lr)

    # Freeze probe - no gradients should flow into it.
    # Save original device so we can restore it after training.
    probe_params = list(probe.parameters())
    probe_original_device = probe_params[0].device if probe_params else device
    probe.eval()
    probe.to(device)
    for p in probe.parameters():
        p.requires_grad = False

    # Get the embedding matrix for computing soft embeddings from perturbed
    # distributions. This creates the gradient path:
    # perturbation_net -> perturbed_dist -> soft_embedding -> probe -> loss
    # Cast to float32 to match the perturbation net's dtype.
    embedding_weight = _get_embedding_weight(
        model, device, hidden_dim, vocab_size
    ).float()

    loss_history: list[float] = []
    kl_history: list[float] = []
    adversarial_history: list[float] = []

    try:
        perturbation_net.train()
        for step in range(config.steps):
            context = train_contexts[step % len(train_contexts)]

            # Forward through frozen LLM - no grad on model params
            with torch.no_grad():
                logits, hidden_state = _forward_with_hidden(model, context)

            # Detach so no gradient flows to LLM.
            # Cast to float32 - model may output bfloat16/float16 but the
            # perturbation net and probe operate in float32 for training stability.
            logits = logits.detach().float().requires_grad_(True)
            hidden_state = hidden_state.detach().float()

            # Apply perturbation (grad flows through perturbation_net)
            perturbed_dist = apply_perturbation(
                logits,
                hidden_state,
                perturbation_net,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            original_dist = torch.softmax(logits, dim=-1).detach()

            # Compute soft embedding: weighted sum of token embeddings by perturbed
            # distribution. This creates a differentiable path from perturbation_net
            # to the probe input. Shape: (hidden_dim,)
            soft_embedding = perturbed_dist @ embedding_weight

            # Probe the soft embedding - probe is frozen (requires_grad=False on
            # its params) so only the perturbation net receives gradients via
            # the perturbed_dist -> soft_embedding -> probe_score path.
            probe_score = torch.sigmoid(probe(soft_embedding))

            # Adversarial loss: probe should output 0 (normal) for stego input.
            adversarial_loss = -torch.log(1.0 - probe_score + 1e-8)

            # KL divergence: original || perturbed (non-negative by Gibbs' inequality)
            kl_div = torch.sum(
                original_dist
                * (
                    torch.log(original_dist.clamp(min=1e-10))
                    - torch.log(perturbed_dist.clamp(min=1e-10))
                )
            )

            loss = adversarial_loss + config.kl_weight * kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            kl_history.append(kl_div.item())
            adversarial_history.append(adversarial_loss.item())

        perturbation_net.eval()
    finally:
        # Restore probe state even if training aborts
        for p in probe.parameters():
            p.requires_grad = True
        probe.to(probe_original_device)

    return AdversarialTrainingResult(
        perturbation=perturbation_net,
        loss_history=loss_history,
        kl_history=kl_history,
        adversarial_history=adversarial_history,
    )


def _get_model_device(model: StegoModel) -> torch.device:
    """Get the device the model is running on."""
    if hasattr(model.model, "device"):
        dev = model.model.device
        if isinstance(dev, torch.device):
            return dev
        return torch.device(dev)
    return torch.device("cpu")


def _get_hidden_dim(model: StegoModel) -> int:
    """Extract hidden dimension from model config."""
    cfg = model.model.config
    # Different architectures use different attribute names
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise ValueError(f"Cannot determine hidden_dim from model config: {cfg}")


def _get_embedding_weight(
    model: StegoModel,
    device: torch.device,
    hidden_dim: int,
    vocab_size: int,
) -> Tensor:
    """Get or create the token embedding matrix for soft-embedding computation.

    If the model exposes get_input_embeddings(), use its weight matrix.
    Otherwise create a frozen random projection (sufficient for training
    the perturbation net - the gradient path is what matters, not the
    specific embedding values).
    """
    try:
        embed_layer = model.model.get_input_embeddings()
        if embed_layer is not None and hasattr(embed_layer, "weight"):
            weight = embed_layer.weight.detach().to(device)
            # Ensure shape is (vocab_size, hidden_dim)
            if weight.shape == (vocab_size, hidden_dim):
                return weight
    except (AttributeError, TypeError):
        pass

    # Fallback: fixed random projection. Frozen, so no extra trainable params.
    gen = torch.Generator(device="cpu").manual_seed(0)
    weight = torch.randn(vocab_size, hidden_dim, generator=gen).to(device)
    return weight


def _forward_with_hidden(
    model: StegoModel, input_ids: list[int]
) -> tuple[Tensor, Tensor]:
    """Run model forward pass, returning logits and last hidden state.

    Both outputs are detached from the model's computation graph.
    """
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    if hasattr(model.model, "device"):
        ids_tensor = ids_tensor.to(model.model.device)

    outputs = model.model(ids_tensor, output_hidden_states=True)
    logits = outputs.logits[0, -1, :]

    # Last layer hidden state, last position
    last_hidden = outputs.hidden_states[-1][0, -1, :]
    return logits, last_hidden


class PerturbedStegoChannel:
    """Steganographic channel that applies distribution perturbation for evasion.

    Wraps an ArithmeticStegoChannel but replaces the raw model distribution
    with a perturbed version at both encode and decode time. Both sides use
    identical perturbation so the arithmetic coder sees the same distribution
    and message recovery works.
    """

    def __init__(
        self,
        inner: ArithmeticStegoChannel,
        perturbation: DistributionPerturbation,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        self._inner = inner
        # Default to inner channel's sampling settings
        self._temperature = (
            temperature if temperature is not None else inner._temperature
        )
        self._top_p = top_p if top_p is not None else inner._top_p
        # Access the model from the inner channel
        self._model: StegoModel = inner._model
        # Derive device from actual activations (handles sharded models)
        device = _get_model_device(self._model)
        self._perturbation = perturbation.to(device)
        self._device_resolved = False

    def _get_perturbed_distribution(self, context_ids: list[int]) -> list[float]:
        """Get perturbed probability distribution for the next token.

        Runs the LLM forward, applies the perturbation network in logit space,
        then returns probabilities as a plain list (compatible with ArithmeticEncoder).
        """
        with torch.no_grad():
            logits, hidden_state = _forward_with_hidden(self._model, context_ids)

            # On first call, move perturbation to the actual activation device
            # (handles sharded models where model.device != output device)
            if not self._device_resolved:
                self._perturbation = self._perturbation.to(logits.device)
                self._device_resolved = True

            logits_f = logits.float()
            hidden_f = hidden_state.float()

            # Match base channel's greedy semantics: temperature<=0 means argmax
            if self._temperature <= 0:
                probs = torch.zeros_like(logits_f)
                probs[logits_f.argmax()] = 1.0
                return probs.cpu().tolist()

            perturbed = apply_perturbation(
                logits_f,
                hidden_f,
                self._perturbation,
                temperature=self._temperature,
                top_p=self._top_p,
            )
        return perturbed.cpu().tolist()

    def encode(self, secret: str, prompt: str, max_tokens: int = 200) -> StegoOutput:
        """Encode a secret using perturbed distributions."""
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        nonce = None
        key = self._inner._key
        if key is not None:
            secret_bits, nonce = encrypt_bits(secret_bits, key)

        from .coding import ArithmeticEncoder

        encoder = ArithmeticEncoder(secret_bits)
        context_ids = list(self._model.tokenize(prompt))
        tokens: list[int] = []

        for _ in range(max_tokens):
            dist = self._get_perturbed_distribution(context_ids)
            idx = encoder.encode_step(dist)
            tokens.append(idx)
            context_ids.append(idx)
            if encoder.bits_consumed >= len(secret_bits):
                break

        text = self._model.detokenize(tokens)
        # Cap at len(secret_bits): when encoding succeeds, bits_consumed may
        # overshoot (it's an approximation). When max_tokens exhausts first,
        # bits_consumed < len(secret_bits) reflects the actual partial encoding.
        return StegoOutput(
            text=text,
            tokens=tokens,
            bits_encoded=min(encoder.bits_consumed, len(secret_bits)),
            tokens_generated=len(tokens),
            nonce=nonce,
        )

    def decode(
        self, text: str, prompt: str, num_bits: int, nonce: bytes | None = None
    ) -> str:
        """Decode a secret using the same perturbed distributions."""
        stego_ids = self._model.tokenize(text)
        context_ids = list(self._model.tokenize(prompt))

        distributions: list[list[float]] = []
        for token_id in stego_ids:
            dist = self._get_perturbed_distribution(context_ids)
            distributions.append(dist)
            context_ids.append(token_id)

        recovered_bits = decode_message(stego_ids, distributions, num_bits)
        key = self._inner._key
        if key is not None and nonce is not None:
            recovered_bits = decrypt_bits(recovered_bits, key, nonce)
        return recovered_bits
