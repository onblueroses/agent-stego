"""OpenAI-compatible client with loopback defaults and explicit remote consent."""

import os
import re
import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from .remote_policy import RemoteEgressError, RemoteEgressPolicy, require_remote_egress

DEFAULT_MODEL = "ollama/qwen2.5:3b"
LOCAL_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
_CLIENT_POLICY_MARKER = object()


@dataclass
class CompletionResult:
    tool_calls: list[dict]
    content: str | None
    model: str
    usage: dict
    latency_ms: float


@dataclass(frozen=True)
class _ClientBinding:
    model: str
    base_url: str
    remote_policy: RemoteEgressPolicy | None


def get_client(
    model: str = "",
    *,
    remote_policy: RemoteEgressPolicy | None = None,
) -> OpenAI:
    """Build a loopback client, or a policy-bound OpenRouter client."""
    selected_model = model or DEFAULT_MODEL
    _validate_model_id(selected_model)
    if selected_model.startswith("ollama/"):
        if remote_policy is not None:
            raise RemoteEgressError(
                "a remote policy cannot be attached to a loopback model"
            )
        return _build_client(
            base_url=LOCAL_OLLAMA_BASE_URL,
            api_key="ollama",
            binding=_ClientBinding(
                model=selected_model,
                base_url=LOCAL_OLLAMA_BASE_URL,
                remote_policy=None,
            ),
        )

    _authorize_openrouter(remote_policy)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required after explicit remote authorization"
        )
    return _build_client(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        binding=_ClientBinding(
            model=selected_model,
            base_url=OPENROUTER_BASE_URL,
            remote_policy=remote_policy,
        ),
    )


def chat_completion(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
) -> CompletionResult:
    """Send one request through the destination bound by ``get_client``."""
    binding = _authorize_client_destination(client)
    selected_model = binding.model if model is None else model
    _validate_model_id(selected_model)
    if selected_model != binding.model:
        raise RemoteEgressError(
            "completion model must exactly match the client-bound model"
        )

    start = time.monotonic()
    api_model = selected_model.removeprefix("ollama/")
    kwargs = {"model": api_model, "messages": messages}
    if tools:
        kwargs["tools"] = tools

    # The SDK transport is configured for zero automatic retries. A failed
    # request therefore cannot multiply spend or resend a payload silently.
    response = client.chat.completions.create(**kwargs)
    elapsed = (time.monotonic() - start) * 1000  # seconds-to-ms unit conversion

    if not response.choices:
        return CompletionResult(
            tool_calls=[],
            content=None,
            model=selected_model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            latency_ms=elapsed,
        )

    choice = response.choices[0]
    tool_calls = []
    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )

    return CompletionResult(
        tool_calls=tool_calls,
        content=choice.message.content,
        model=selected_model,
        usage={
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": (
                response.usage.completion_tokens if response.usage else 0
            ),
        },
        latency_ms=elapsed,
    )


def _authorize_openrouter(policy: RemoteEgressPolicy | None) -> None:
    require_remote_egress(
        policy=policy,
        provider="openrouter",
        destination=OPENROUTER_BASE_URL,
        action="send synthetic experiment prompts to OpenRouter",
    )


def _authorize_client_destination(client: OpenAI) -> _ClientBinding:
    marker = getattr(client, "_agent_stego_policy_marker", None)
    if (
        not isinstance(marker, tuple)
        or len(marker) != 2
        or marker[0] is not _CLIENT_POLICY_MARKER
        or not isinstance(marker[1], _ClientBinding)
    ):
        raise RemoteEgressError(
            "completion clients must be constructed by get_client so transport "
            "policy cannot be bypassed"
        )

    binding = marker[1]
    base_url = str(client.base_url).rstrip("/")
    if base_url != binding.base_url:
        raise RemoteEgressError("completion client destination changed after binding")
    if base_url == LOCAL_OLLAMA_BASE_URL and binding.remote_policy is None:
        return binding
    if base_url == OPENROUTER_BASE_URL and binding.remote_policy is not None:
        _authorize_openrouter(binding.remote_policy)
        return binding
    raise RemoteEgressError(
        "completion client destination is neither exact loopback Ollama nor "
        "the explicitly authorized OpenRouter endpoint"
    )


def _build_client(
    *,
    base_url: str,
    api_key: str,
    binding: _ClientBinding,
) -> OpenAI:
    transport = httpx.Client(follow_redirects=False, trust_env=False)
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=transport,
        max_retries=0,
    )
    setattr(client, "_agent_stego_policy_marker", (_CLIENT_POLICY_MARKER, binding))
    return client


def _validate_model_id(model: object) -> None:
    if not isinstance(model, str) or _MODEL_PATTERN.fullmatch(model) is None:
        raise ValueError("model must use the canonical model-id grammar")
