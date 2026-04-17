"""HuggingFace model wrapper for token-level steganography.

Provides access to raw logits/probability distributions over the
full vocabulary, which ollama doesn't expose.
"""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class StegoModel:
    """Wraps a HuggingFace causal LM for steganographic sampling."""

    def __init__(self, model_name: str, dtype: torch.dtype | None = None) -> None:
        self.model_name = model_name
        self._dtype = dtype
        # typed as Any because transformers' Auto classes lack proper stubs
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        """Load the model and tokenizer. Call before any other method."""
        kwargs: dict = {"torch_dtype": self._dtype} if self._dtype else {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("Call load() before using the model")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("Call load() before using the tokenizer")
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize_chat(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> list[int]:
        """Tokenize a chat conversation using the model's chat template.

        Avoids string round-trip drift by going directly to token IDs.
        """
        result = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt
        )
        # Some tokenizers return BatchEncoding; extract input_ids
        if hasattr(result, "input_ids"):
            return result.input_ids
        return result

    def detokenize(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def get_logits(self, input_ids: list[int]) -> torch.Tensor:
        """Get raw logits for the next token. Gradient-compatible.

        Unlike get_distribution, this returns a torch Tensor (not a list)
        and does NOT apply no_grad, so gradients can flow through
        perturbation networks applied to the output.

        Returns:
            Logits tensor of shape (vocab_size,) on the model's device.
        """
        ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        if hasattr(self.model, "device"):
            ids_tensor = ids_tensor.to(self.model.device)
        outputs = self.model(ids_tensor)
        return outputs.logits[0, -1, :]

    @torch.no_grad()
    def get_distribution(
        self,
        input_ids: list[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> list[float] | tuple[list[float], Any]:
        """Get the probability distribution over next tokens.

        Args:
            input_ids: Token IDs for the context so far.
            temperature: Softmax temperature. 1.0 = no change. Lower = sharper.
            top_p: Nucleus sampling threshold. 1.0 = full vocab. 0.9 = top 90%.
            past_key_values: Optional KV-cache from a previous call. When provided,
                only the LAST token in input_ids is fed to the model (the rest
                are already cached).
            use_cache: If True, return (distribution, past_key_values) tuple.
                If False (default), return just the distribution list.

        Returns:
            List of probabilities when use_cache is False (backward compatible).
            Tuple of (probabilities, past_key_values) when use_cache is True.
        """

        if past_key_values is not None:
            # Only feed the last token - the rest are in the cache
            ids_tensor = torch.tensor([[input_ids[-1]]], dtype=torch.long)
        else:
            ids_tensor = torch.tensor([input_ids], dtype=torch.long)

        if hasattr(self.model, "device"):
            ids_tensor = ids_tensor.to(self.model.device)

        outputs = self.model(
            ids_tensor,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[0, -1, :]  # last position

        # Apply temperature scaling before softmax
        if temperature <= 0:
            # temperature=0 means greedy: put all mass on the argmax token
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
            return probs.cpu().tolist()
        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=0)

        # Apply top-p (nucleus) filtering: zero out tokens outside the nucleus
        if top_p <= 0 or top_p > 1.0:
            top_p = 1.0  # invalid values fall back to full vocab
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            # Find cutoff: keep tokens until cumulative prob exceeds top_p
            # Always keep at least the top token
            cutoff_mask = cumsum - sorted_probs >= top_p
            sorted_probs[cutoff_mask] = 0.0
            # Scatter back to original positions
            probs = torch.zeros_like(probs)
            probs.scatter_(0, sorted_indices, sorted_probs)
            # Renormalize
            total = probs.sum()
            if total > 0:
                probs = probs / total

        prob_list = probs.cpu().tolist()
        if use_cache:
            return prob_list, outputs.past_key_values
        return prob_list
