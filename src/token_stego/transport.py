"""Lossless token-to-text transport contract for token steganography."""

from .model import StegoModel


def text_transport(model: StegoModel, tokens: list[int]) -> str:
    """Serialize tokens only when text observation preserves every token ID."""
    text = model.detokenize(tokens)
    if model.tokenize(text) != tokens:
        raise ValueError(
            "Generated token IDs do not survive the declared text transport"
        )
    return text
