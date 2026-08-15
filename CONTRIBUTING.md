# Contributing

## Set up the repository

```bash
git clone https://github.com/onblueroses/agent-stego.git
cd agent-stego
uv sync --extra dev --extra token
```

## Run the release gates

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pyright src/ experiments/
uv run pytest tests/ -q
```

The tests use mocks and fixed fixtures, so they need no GPU or API key. The
`token` extra supplies PyTorch and Transformers for modules imported by the full
suite.

## Add a channel

Implement `BaseChannel` from `src/channels/base.py`, add a versioned synthetic
prompt under `prompts/` when the channel needs one, and cover the encoder,
decoder, malformed input, and incomplete-delivery paths in `tests/`.

Keep one mechanism per pull request. A renamed carrier, new model, or parameter
change is not a new channel unless its causal encoding mechanism differs.

## Add an experiment

Experiment drivers belong in `experiments/`. Use the shared client for model
calls; direct provider SDK clients and automatic `.env` loading are not accepted.
Local Ollama is the default. Remote inference must present an exact, unexpired
`RemoteEgressPolicy` at the network boundary, and every payload must be synthetic.

Raw output goes in ignored `findings/`. Before publishing a result, record the
source and model revisions, payload classification, split unit, seeds,
uncertainty method, artifact hashes, and regeneration command. Choose sample
counts from a stated effect size and power analysis. If the run cannot support a
detection limit and a positive control, report it as exploratory.

## Code and prose

Use Python 3.12 or newer, type public interfaces, and keep validation at trust
boundaries. Public docs should distinguish tested mechanics, historical
observations, hypotheses, and missing evidence. Do not turn a convenient scalar
into a threshold unless a specification, measurement, or cited method chose it.

Never commit credentials, private prompts, host details, raw model traces, or
agent-session artifacts. Use GitHub's private vulnerability-reporting channel
for security issues.
