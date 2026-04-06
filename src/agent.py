import time
from dataclasses import dataclass

from .client import CompletionResult, chat_completion, get_client, DEFAULT_MODEL
from .harness import ToolHarness, ToolTrace, create_default_harness


@dataclass
class AgentRun:
    system_prompt: str
    user_message: str
    model: str
    traces: list[ToolTrace]
    messages: list[dict]
    completions: list[CompletionResult]
    total_time: float
    iterations: int


def run_agent(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    harness: ToolHarness | None = None,
    max_iterations: int = 15,
) -> AgentRun:
    client = get_client()
    harness = harness or create_default_harness()
    harness.reset()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    completions = []
    start = time.monotonic()

    for i in range(max_iterations):
        result = chat_completion(
            client, messages, tools=harness.get_tool_schemas(), model=model
        )
        completions.append(result)

        if not result.tool_calls:
            break

        # Build assistant message with tool calls
        assistant_msg = {"role": "assistant", "content": result.content}
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in result.tool_calls
        ]
        messages.append(assistant_msg)

        # Execute each tool call and add results
        for tc in result.tool_calls:
            tool_response = harness.call(tc["name"], tc["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_response,
                }
            )

    elapsed = time.monotonic() - start

    return AgentRun(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        traces=list(harness.traces),
        messages=messages,
        completions=completions,
        total_time=elapsed,
        iterations=len(completions),
    )
