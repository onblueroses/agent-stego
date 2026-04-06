import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"


@dataclass
class CompletionResult:
    tool_calls: list[dict]
    content: str | None
    model: str
    usage: dict
    latency_ms: float


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def chat_completion(
    client: OpenAI,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
) -> CompletionResult:
    start = time.monotonic()
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools

    response = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2**attempt
            print(f"  [retry {attempt + 1}/3 after {wait}s: {e}]")
            time.sleep(wait)

    assert response is not None
    elapsed = (time.monotonic() - start) * 1000

    choice = response.choices[0]
    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            )

    return CompletionResult(
        tool_calls=tool_calls,
        content=choice.message.content,
        model=model,
        usage={
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else 0,
        },
        latency_ms=elapsed,
    )
