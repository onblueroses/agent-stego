# agent-stego

agent-stego is research code for covert channels in the text and tool traces of AI agents. The
detectors and the failure cases are next to the channel implementations.

## What it does

If an agent can choose among several plausible tokens or tool actions, how much of that choice can
carry a second message? This repository makes that question executable. It looks for the boundary
between a channel that works in a toy decoder and a channel that survives transport, realistic
controls and a defender who knows what to look for.

This is an alpha research repository. The reports under `analysis/` and `studies/` are a lab
notebook, not a benchmark suite. Several historical results are explicitly invalidated. I keep them
with their limits visible: [Evidence boundary](docs/details.md#evidence-boundary).

## Carrier surfaces

| Surface | Where the message goes | Needs a model |
|---|---|---|
| Tool-call channels | Which fake tool, call order, arguments, timing, state between turns | No (canned harness) |
| Document and text carriers | Unicode, whitespace, capitalization, JSON key order, headings, file content | Most run offline |
| Token channels | The next-token distribution of a language model (arithmetic coder baseline) | Yes (`token` extra) |

Detectors are next to the channels that they examine. The harness tools do not do real
filesystem, shell or web operations. Full notes on each surface:
[docs/details.md](docs/details.md#the-three-carrier-surfaces).

## Example

The channel classes are ordinary Python objects. This example makes the instructions for a
lookup-table carrier. It does not make a network request.

```python
from src.channels.table import TableChannel

channel = TableChannel(mode="path")
prompt = channel.encode("HI")
print(prompt)
```

## How to install and run

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). The core tool-call channels and
their tests run without a model, GPU or API key.

```bash
git clone https://github.com/onblueroses/agent-stego.git
cd agent-stego
uv sync --extra dev
uv run pytest tests/test_channels.py tests/test_experiments.py -q
```

Token-level experiments need the optional model dependencies:

```bash
uv sync --extra dev --extra token
uv run pytest tests/ -q
```

The default model endpoint is local Ollama on loopback. Remote calls need an explicit
`RemoteEgressPolicy`. An API key alone is not authorization. Use only synthetic payloads. Read the
[safety boundary](docs/details.md#safety-boundary) and [SECURITY.md](SECURITY.md).

## Layout

```text
src/channels/      tool-call and text-carrier channels
src/token_stego/   token coder, transport, probes and detectors
experiments/       standalone research drivers
analysis/          historical reports and limitations
studies/           bounded historical study notes
tests/             unit and contract tests
docs/              details: evidence rules, safety, full map, dev gates
```

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you add a channel or an experimental claim. The
license is [MIT](LICENSE).
