"""Tests for CovertConversation round-trip recovery using mock model."""

from unittest.mock import MagicMock

import torch

from src.token_stego.conversation import CovertConversation


VOCAB_SIZE = 100


def _make_mock_model(vocab_size: int = VOCAB_SIZE) -> MagicMock:
    """Create a mock StegoModel for conversation testing.

    Distribution is deterministic and keyed on context length, so encode
    and decode see identical distributions for the same context.
    """
    model = MagicMock()
    model.vocab_size = vocab_size
    model.model_name = "mock"

    # Mock tokenizer with eos_token_id
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    model.tokenizer = tokenizer

    # tokenize_chat: return deterministic IDs based on message count
    # Each message adds 5 tokens to context
    def mock_tokenize_chat(messages, add_generation_prompt=True):
        base = [10, 20, 30]  # system prompt tokens
        for i, _msg in enumerate(messages[1:]):  # skip system
            base.extend([40 + i * 3, 41 + i * 3, 42 + i * 3, 43 + i * 3, 44 + i * 3])
        if add_generation_prompt:
            base.append(50)
        return base

    model.tokenize_chat = mock_tokenize_chat

    def mock_decode(ids, skip_special_tokens=True):
        # Encode the token IDs into the "text" so retokenization can recover them
        key = ",".join(str(i) for i in ids)
        return f"TOKENS:{key}"

    tokenizer.decode = mock_decode

    def mock_tokenize(text):
        if text.startswith("TOKENS:"):
            return [int(x) for x in text[7:].split(",")]
        return [1, 2, 3]

    model.tokenize = mock_tokenize

    # Deterministic distribution keyed on context length
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


class TestCovertConversationRoundTrip:
    """Step 1.5: Alice sends 'HI' to Bob, Bob recovers it exactly."""

    def test_alice_sends_hi_bob_recovers(self) -> None:
        torch.manual_seed(42)
        model = _make_mock_model()

        conv = CovertConversation(
            model=model,
            cover_topic="Discuss computing",
            temperature=1.0,
            top_p=1.0,
        )

        result = conv.run(
            alice_secret="HI",
            bob_secret="OK",
            num_turns=4,
            max_tokens_per_turn=50,
        )

        # Alice's message should be recovered by Bob
        assert result.alice_recovered_by_bob == "HI", (
            f"Expected 'HI', got '{result.alice_recovered_by_bob}'"
        )

    def test_bob_secret_recovered_by_alice(self) -> None:
        torch.manual_seed(42)
        model = _make_mock_model()

        conv = CovertConversation(
            model=model,
            cover_topic="Discuss computing",
            temperature=1.0,
            top_p=1.0,
        )

        result = conv.run(
            alice_secret="HI",
            bob_secret="OK",
            num_turns=4,
            max_tokens_per_turn=50,
        )

        assert result.bob_recovered_by_alice == "OK", (
            f"Expected 'OK', got '{result.bob_recovered_by_alice}'"
        )

    def test_overall_exact_recovery(self) -> None:
        torch.manual_seed(42)
        model = _make_mock_model()

        conv = CovertConversation(
            model=model,
            cover_topic="Discuss computing",
            temperature=1.0,
            top_p=1.0,
        )

        result = conv.run(
            alice_secret="HI",
            bob_secret="OK",
            num_turns=4,
            max_tokens_per_turn=50,
        )

        assert result.overall_exact, (
            f"Expected exact recovery. "
            f"Alice recovered: '{result.alice_recovered_by_bob}', "
            f"Bob recovered: '{result.bob_recovered_by_alice}'"
        )

    def test_per_turn_bits_encoded(self) -> None:
        """Each turn should encode some bits."""
        torch.manual_seed(42)
        model = _make_mock_model()

        conv = CovertConversation(
            model=model,
            cover_topic="Discuss computing",
            temperature=1.0,
            top_p=1.0,
        )

        result = conv.run(
            alice_secret="HI",
            bob_secret="OK",
            num_turns=4,
            max_tokens_per_turn=50,
        )

        assert result.total_bits_exchanged > 0
        # At least the first turn should encode bits
        assert result.turns[0].bits_encoded > 0

    def test_with_encryption(self) -> None:
        """Round-trip works with encryption enabled."""
        torch.manual_seed(42)
        model = _make_mock_model()

        conv = CovertConversation(
            model=model,
            cover_topic="Discuss computing",
            key=b"test-conversation-key",
            temperature=1.0,
            top_p=1.0,
        )

        result = conv.run(
            alice_secret="HI",
            bob_secret="OK",
            num_turns=4,
            max_tokens_per_turn=50,
        )

        assert result.overall_exact, (
            f"Encrypted recovery failed. "
            f"Alice: '{result.alice_recovered_by_bob}', "
            f"Bob: '{result.bob_recovered_by_alice}'"
        )

    def test_turn_roles_alternate(self) -> None:
        """Turns should alternate alice/bob roles."""
        torch.manual_seed(42)
        model = _make_mock_model()

        conv = CovertConversation(
            model=model,
            cover_topic="Discuss computing",
            temperature=1.0,
            top_p=1.0,
        )

        result = conv.run(
            alice_secret="A",
            bob_secret="B",
            num_turns=4,
            max_tokens_per_turn=50,
        )

        roles = [t.role for t in result.turns]
        assert roles == ["alice", "bob", "alice", "bob"]
