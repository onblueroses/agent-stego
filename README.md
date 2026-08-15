# agent-stego

Research code for covert channels in AI-agent text and tool traces, with the
detectors and failure cases kept beside the channel implementations.

I built this repository to make a slippery security question executable: if an
agent can choose among several plausible tokens or tool actions, how much of that
choice can carry a second message? The interesting part isn't inventing one more
encoding trick. It is finding the boundary between a channel that works in a toy
decoder and one that survives transport, realistic controls, and a defender who
knows what to look for.

This is an alpha research repository. The code contains tested channel mechanics
and experiment-design safeguards. The numerical reports under `analysis/` are a
lab notebook, not a benchmark suite. Several historical results are explicitly
invalidated because the original runs leaked prompt groups, counted incomplete
payloads as capacity, or lack enough artifact lineage for reproduction.

## Start here

The core tool-call channels and their tests run without a model, GPU, or API key:

```bash
git clone https://github.com/onblueroses/agent-stego.git
cd agent-stego
uv sync --extra dev
uv run pytest tests/test_channels.py tests/test_experiments.py -q
```

The channel classes are ordinary Python objects. This example builds the
instructions for a lookup-table carrier without making a network request:

```python
from src.channels.table import TableChannel

channel = TableChannel(mode="path")
prompt = channel.encode("HI")
print(prompt)
```

Token-level experiments need the optional model dependencies:

```bash
uv sync --extra dev --extra token
uv run pytest tests/ -q
```

Python 3.12 is the supported minimum because that is the lower bound declared in
`pyproject.toml` and exercised in CI.

## What is here

The experiments span three carrier surfaces.

Tool-call channels encode information in action choices: which fake tool is
selected, the order of calls, arguments, timing, or state carried between turns.
The default harness returns canned responses. Its tool names resemble filesystem,
shell, and web operations, but the handlers do not perform those operations.

Document and text carriers encode information in representations an agent reads
or writes: Unicode code points, whitespace, capitalization, JSON key order,
headings, and composed file content. Most of these baselines run offline; whether
a real model and transport preserve a given representation is a separate
measurement.

Token channels use a language model's next-token distribution. The current
arithmetic coder is a finite-message target-point baseline. It is useful for
studying exact transport and detection, but it is not claimed to preserve the
model distribution or instantiate a published indistinguishability proof.

Detectors live next to the channels they examine. The evaluation helpers enforce
group-disjoint splits, reject contradictory duplicate observations, retain split
manifests, and report uncertainty without turning an arbitrary cutoff into a
scientific verdict.

## Evidence boundary

I keep historical reports because failed and superseded runs are useful when
their limits are visible. They do not all describe the current code.

- [analysis/FINDINGS.md](analysis/FINDINGS.md) is the historical tool-channel
  notebook and carries a release-status warning.
- [analysis/token_likelihood_results.md](analysis/token_likelihood_results.md)
  records early token-coder observations and their invalidation.
- [analysis/rerun_2026-05-02_lengthmatch.md](analysis/rerun_2026-05-02_lengthmatch.md)
  explains why length matching alone did not repair the activation-probe study.
- [analysis/capacity_transfer_results.md](analysis/capacity_transfer_results.md)
  and [analysis/coupled_probe_results.md](analysis/coupled_probe_results.md) are
  retained for provenance, not as current measurements.

A fresh result is not ready for a public claim unless it records the source and
model revisions, payload classification, split unit, seeds, uncertainty method,
raw-artifact lineage, and an exact regeneration command. A negative result also
needs a positive control and a detection limit. Without those, the honest wording
is "not yet made to work."

## Safety boundary

Local Ollama on literal loopback is the default model endpoint. The shared client
does not inherit proxy settings, follow redirects, or retry a request
automatically. Remote OpenRouter calls require an explicit, unexpired
`RemoteEgressPolicy` bound to the exact provider, destination, model action, and
synthetic-payload confirmation. An API key alone is not authorization.

Use only synthetic payloads and infrastructure you own or are authorized to use.
Do not put credentials, personal data, or private prompts in `SECRET`, `KEY`, model
messages, filenames, logs, or reports. Raw experiment output belongs in the
ignored `findings/` directory until it has been reviewed and redacted.

See [SECURITY.md](SECURITY.md) for the threat boundary and private reporting path.

## Repository map

```text
src/
  agent.py                 local agent loop
  harness.py               canned tool implementations and trace capture
  client.py                loopback client and policy-bound remote client
  remote_policy.py         explicit remote-egress capability
  channels/                tool-call and text-carrier implementations
  token_stego/             token coder, transport, probes, and detectors
experiments/               standalone research drivers
analysis/                  historical reports and limitations
prompts/                   versioned synthetic encoder prompts
scripts/                   local measurement and verification helpers
studies/                   bounded historical study notes
tests/                     unit and contract tests
```

## Development gates

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pyright src/ experiments/
uv run pytest tests/ -q
```

CI also audits the complete exported lock graph for known advisories and scans the
Git history for credential signatures. These checks have detection limits: an
advisory scanner only knows its current database, and a secret scanner cannot
recognize arbitrary operational metadata. Review filenames, reports, and history
as well.

## Related work

- [PNR-KE: Undetectable Conversations Between AI Agents](https://arxiv.org/abs/2604.04757)
- [Efficient Provably Secure Linguistic Steganography via Range Coding](https://arxiv.org/abs/2604.08052)
- [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion](https://arxiv.org/abs/2410.03768)
- [Perfectly Secure Steganography Using Minimum Entropy Coupling](https://arxiv.org/abs/2210.14889)

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before adding a channel or experimental
claim. The code is available under the [MIT License](LICENSE).
