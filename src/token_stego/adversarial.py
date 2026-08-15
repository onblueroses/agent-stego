"""Perturbed token channel plus a retired adversarial-training interface.

The former trainer optimized an activation probe on a soft token embedding,
not on the hidden-state feature the probe observes after model execution. That
surrogate was causally invalid, so the public trainer now fails closed. The
mechanical PerturbedStegoChannel remains available for controlled tests of a
provided perturbation network.
"""

import os
from dataclasses import dataclass

import torch
from torch import Tensor

from .activation_probe import ActivationProbe
from .base import StegoOutput
from .channel import DEFAULT_MAX_TOKENS, ArithmeticStegoChannel
from .coding import decode_message
from .crypto import decrypt_bits, encrypt_bits
from .model import StegoModel
from .perturbation import (
    DistributionPerturbation,
    apply_perturbation,
    validate_sampling_parameters,
)
from .transport import text_transport

RETIRED_ADVERSARIAL_TRAINING_ERROR = (
    "Adversarial training is retired: the former soft-embedding surrogate "
    "did not optimize the probe's causal hidden-state observation"
)


@dataclass
class AdversarialTrainingConfig:
    """Retained only so callers receive the explicit retirement error."""

    steps: int = 200
    lr: float = 1e-3
    kl_weight: float = 0.1
    temperature: float = 1.0
    top_p: float = 1.0


def train_perturbation(
    model: StegoModel,
    probe: ActivationProbe,
    train_contexts: list[list[int]],
    config: AdversarialTrainingConfig | None = None,
) -> None:
    """Reject the retired soft-embedding surrogate.

    A replacement must backpropagate through the actual model rollout into the
    exact frozen probe feature extractor, then validate evasion on discretely
    generated, held-out activations. Returning weights before those conditions
    hold would recreate the invalid historical result.
    """
    del model, probe, train_contexts, config
    raise RuntimeError(RETIRED_ADVERSARIAL_TRAINING_ERROR)


def _get_model_device(model: StegoModel) -> torch.device:
    """Get the device the model is running on."""
    if hasattr(model.model, "device"):
        dev = model.model.device
        if isinstance(dev, torch.device):
            return dev
        return torch.device(dev)
    return torch.device("cpu")


def _forward_with_hidden(
    model: StegoModel, input_ids: list[int]
) -> tuple[Tensor, Tensor]:
    """Run model forward pass, returning logits and last hidden state.

    The caller chooses the gradient context. Perturbed inference uses no-grad.
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
        validate_sampling_parameters(self._temperature, self._top_p)
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
        validate_sampling_parameters(self._temperature, self._top_p)
        with torch.no_grad():
            logits, hidden_state = _forward_with_hidden(self._model, context_ids)

            # On first call, move perturbation to the actual activation device
            # (handles sharded models where model.device != output device)
            if not self._device_resolved:
                self._perturbation = self._perturbation.to(logits.device)
                self._device_resolved = True

            logits_f = logits.float()
            hidden_f = hidden_state.float()

            # Match base channel's greedy semantics: temperature=0 means argmax
            if self._temperature == 0:
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

    def encode(
        self,
        secret: str,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> StegoOutput:
        """Encode a secret using perturbed distributions."""
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
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
            if encoder.complete:
                break

        text = text_transport(self._model, tokens)
        return StegoOutput(
            text=text,
            tokens=tokens,
            bits_encoded=min(encoder.bits_consumed, len(secret_bits)),
            tokens_generated=len(tokens),
            nonce=nonce,
            complete=encoder.complete,
        )

    def decode(
        self, text: str, prompt: str, num_bits: int, nonce: bytes | None = None
    ) -> str:
        """Decode a secret using the same perturbed distributions."""
        key = self._inner._key
        if key is not None and nonce is None:
            raise ValueError("Key is set but nonce is missing; cannot decrypt.")
        stego_ids = self._model.tokenize(text)
        context_ids = list(self._model.tokenize(prompt))

        distributions: list[list[float]] = []
        for token_id in stego_ids:
            dist = self._get_perturbed_distribution(context_ids)
            distributions.append(dist)
            context_ids.append(token_id)

        recovered_bits = decode_message(stego_ids, distributions, num_bits)
        if key is not None:
            assert nonce is not None
            recovered_bits = decrypt_bits(recovered_bits, key, nonce)
        return recovered_bits


def save_perturbation(
    perturbation: DistributionPerturbation, path: str | os.PathLike[str]
) -> None:
    """Save a trained perturbation network to disk.

    Persists the state_dict along with architecture metadata needed to
    reconstruct the network on load.

    Args:
        perturbation: Trained DistributionPerturbation to save.
        path: File path to write the checkpoint to.
    """
    # Extract architecture dimensions from the network layers
    first_layer = perturbation.net[0]
    last_layer = perturbation.net[-1]
    assert isinstance(first_layer, torch.nn.Linear)
    assert isinstance(last_layer, torch.nn.Linear)

    checkpoint = {
        "state_dict": perturbation.state_dict(),
        "hidden_dim": first_layer.in_features,
        "vocab_size": last_layer.out_features,
        "bottleneck": first_layer.out_features,
    }
    torch.save(checkpoint, path)


def load_perturbation(path: str | os.PathLike[str]) -> DistributionPerturbation:
    """Load a perturbation network from a checkpoint file.

    Reconstructs the DistributionPerturbation architecture from stored
    metadata, then loads the saved weights.

    Args:
        path: File path to load the checkpoint from.

    Returns:
        DistributionPerturbation with restored weights.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "state_dict",
        "hidden_dim",
        "vocab_size",
        "bottleneck",
    }:
        raise ValueError("Perturbation checkpoint schema is invalid")
    dimensions = [
        checkpoint[name] for name in ("hidden_dim", "vocab_size", "bottleneck")
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in dimensions
    ):
        raise ValueError("Perturbation checkpoint dimensions must be positive integers")
    hidden_dim, vocab_size, bottleneck = dimensions

    net = DistributionPerturbation(hidden_dim, vocab_size, bottleneck=bottleneck)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net
