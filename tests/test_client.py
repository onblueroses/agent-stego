"""Tests for client routing and remote policy binding."""

from datetime import UTC, datetime, timedelta

import pytest

import src.client as client_module
from src.client import chat_completion, get_client
from src.remote_policy import RemoteEgressError, RemoteEgressPolicy


def _openrouter_policy() -> RemoteEgressPolicy:
    return RemoteEgressPolicy(
        approval_record_id="test-approval",
        provider="openrouter",
        destination="https://openrouter.ai/api/v1",
        action="send synthetic experiment prompts to OpenRouter",
        # One hour is a test-fixture window, not a production policy default.
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        synthetic_payload_confirmed=True,
    )


def test_ollama_routes_to_literal_loopback() -> None:
    client = get_client("ollama/qwen2.5:3b")
    assert str(client.base_url).startswith("http://127.0.0.1:11434/")
    assert client.max_retries == 0
    assert client._client.follow_redirects is False
    assert client._client._trust_env is False


def test_remote_models_are_blocked_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-used")
    with pytest.raises(RemoteEgressError, match="in-process policy"):
        get_client("meta-llama/llama-3.1-8b-instruct")


def test_missing_api_key_raises_after_explicit_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_client(
            "meta-llama/llama-3.1-8b-instruct",
            remote_policy=_openrouter_policy(),
        )


def test_unknown_prebuilt_client_is_rejected_before_request() -> None:
    class FakeClient:
        base_url = "https://example.com/v1"

    with pytest.raises(RemoteEgressError, match="constructed by get_client"):
        chat_completion(FakeClient(), [], model="model")  # type: ignore[arg-type]


def test_completion_rejects_model_different_from_binding() -> None:
    client = get_client("ollama/qwen2.5:3b")
    with pytest.raises(RemoteEgressError, match="client-bound model"):
        chat_completion(client, [], model="ollama/qwen2.5:7b")


def test_completion_rejects_destination_changed_after_binding() -> None:
    client = get_client("ollama/qwen2.5:3b")
    client.base_url = "https://example.com/v1"
    with pytest.raises(RemoteEgressError, match="destination changed"):
        chat_completion(client, [])


def test_transport_disables_proxy_redirects_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = object()
    captured: dict[str, object] = {}

    def fake_http_client(**kwargs: object) -> object:
        captured["http"] = kwargs
        return transport

    class FakeOpenAI:
        base_url = client_module.LOCAL_OLLAMA_BASE_URL

        def __init__(self, **kwargs: object) -> None:
            captured["openai"] = kwargs

    monkeypatch.setattr(client_module.httpx, "Client", fake_http_client)
    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)
    get_client("ollama/qwen2.5:3b")

    assert captured["http"] == {"follow_redirects": False, "trust_env": False}
    assert captured["openai"] == {
        "base_url": client_module.LOCAL_OLLAMA_BASE_URL,
        "api_key": "ollama",
        "http_client": transport,
        "max_retries": 0,
    }
