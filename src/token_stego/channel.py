"""Arithmetic coding steganographic channel using HuggingFace models.

Encodes synthetic messages into generated text by choosing among tokens from
model probability distributions. Naturalness is an empirical outcome to
measure, not a property guaranteed by this sampler.
"""

import math

from .base import StegoOutput, TokenStegoChannel, TokenStegoMetrics
from .coding import InsufficientCapacityError, decode_message
from .crypto import decrypt_bits, encrypt_bits
from .model import StegoModel
from .transport import text_transport

# Historical experiment drivers use this generation ceiling. It is a replay
# default, not a measured capacity requirement; new studies should override it.
DEFAULT_MAX_TOKENS = 200


class ArithmeticStegoChannel(TokenStegoChannel):
    """Steganographic channel using arithmetic coding over token distributions.

    When a key is provided, message bits are encrypted before encoding
    and decrypted after decoding. This scrambles the arithmetic coder's target
    point; it does not by itself establish distributional indistinguishability.
    """

    def __init__(
        self,
        model: StegoModel,
        key: bytes | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self._model = model
        self._key = key
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be finite and in the interval (0, 1]")
        self._temperature = temperature
        self._top_p = top_p

    def encode(
        self,
        secret: str,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> StegoOutput:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        nonce = None
        if self._key is not None:
            secret_bits, nonce = encrypt_bits(secret_bits, self._key)

        # Build context incrementally - each step sees all prior tokens
        context_ids = list(self._model.tokenize(prompt))

        # Generate tokens one at a time, building distributions incrementally
        # so encode and decode see identical distributions
        from .coding import ArithmeticEncoder

        encoder = ArithmeticEncoder(secret_bits)
        tokens: list[int] = []
        past_kv = None

        for _ in range(max_tokens):
            dist, past_kv = self._model.get_distribution(
                context_ids,
                self._temperature,
                self._top_p,
                past_key_values=past_kv,
                use_cache=True,
            )
            idx = encoder.encode_step(dist)
            tokens.append(idx)
            context_ids.append(idx)
            # Stop when we've encoded enough bits
            if encoder.complete:
                break

        # Build the generated text - keep special tokens to preserve round-trip
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
        if self._key is not None and nonce is None:
            raise ValueError("Key is set but nonce is missing; cannot decrypt.")

        # Tokenize the stego text
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        # Replay: for each token, recompute the distribution from context
        distributions: list[list[float]] = []
        current_context = list(context_ids)
        past_kv = None
        for token_id in stego_ids:
            dist, past_kv = self._model.get_distribution(
                current_context,
                self._temperature,
                self._top_p,
                past_key_values=past_kv,
                use_cache=True,
            )
            distributions.append(dist)
            current_context.append(token_id)

        # Map token IDs to indices in the vocabulary
        # The distribution is over the full vocab, so token_id IS the index
        token_indices = stego_ids

        recovered_bits = decode_message(token_indices, distributions, num_bits)
        if self._key is not None:
            assert nonce is not None
            recovered_bits = decrypt_bits(recovered_bits, self._key, nonce)
        return recovered_bits

    def measure(
        self, secret: str, text: str, prompt: str, nonce: bytes | None = None
    ) -> TokenStegoMetrics:
        if self._key is not None and nonce is None:
            raise ValueError(
                "Key is set but nonce is missing; cannot measure recovery."
            )

        secret_bits = "".join(format(b, "08b") for b in secret.encode("ascii"))
        num_bits = len(secret_bits)

        # Decode
        stego_ids = self._model.tokenize(text)
        context_ids = self._model.tokenize(prompt)

        distributions: list[list[float]] = []
        log_probs: list[float] = []
        current_context = list(context_ids)
        past_kv = None

        for token_id in stego_ids:
            dist, past_kv = self._model.get_distribution(
                current_context,
                self._temperature,
                self._top_p,
                past_key_values=past_kv,
                use_cache=True,
            )
            distributions.append(dist)
            # Track log probability of each chosen token
            prob = dist[token_id]
            log_probs.append(math.log(prob) if prob > 0 else float("-inf"))
            current_context.append(token_id)

        # Recover message
        try:
            recovered_bits = decode_message(stego_ids, distributions, num_bits)
        except InsufficientCapacityError:
            recovered_bits = ""
        if self._key is not None and recovered_bits:
            assert nonce is not None
            recovered_bits = decrypt_bits(recovered_bits, self._key, nonce)

        # Bit error rate
        errors = sum(a != b for a, b in zip(secret_bits, recovered_bits))
        errors += abs(len(secret_bits) - len(recovered_bits))
        total = max(len(secret_bits), len(recovered_bits))
        ber = errors / total if total > 0 else 1.0

        # Perplexity: exp(mean(-log_prob))
        if log_probs:
            avg_neg_log = -sum(log_probs) / len(log_probs)
            try:
                perplexity = math.exp(avg_neg_log)
            except OverflowError:
                perplexity = float("inf")
        else:
            perplexity = float("inf")

        # Bits per token
        exact_recovery = recovered_bits == secret_bits
        bpt = num_bits / len(stego_ids) if stego_ids and exact_recovery else 0.0

        return TokenStegoMetrics(
            bit_error_rate=ber,
            bits_per_token=bpt,
            perplexity=perplexity,
            kl_divergence=None,
            exact_recovery=exact_recovery,
            tokens_generated=len(stego_ids),
        )
