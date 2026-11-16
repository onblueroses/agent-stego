# agent-stego in detail

This file keeps the longer text from the old README. The [README](../README.md) gives the short form.

## Why this repository exists

I built this repository to make a difficult security question executable. If an agent can choose
among several plausible tokens or tool actions, how much of that choice can carry a second message?

The interesting part is not one more encoding trick. It is the boundary between two kinds of
channel:

- a channel that works in a toy decoder
- a channel that survives transport, realistic controls and a defender who knows what to look for

## Status

This is an alpha research repository. The code contains tested channel mechanics and safeguards for
experiment design. The numerical reports under `analysis/` and `studies/` are a lab notebook, not a
benchmark suite.

Several historical results are explicitly invalidated. The original runs had one or more of these
problems:

- They leaked prompt groups.
- They counted incomplete payloads as capacity.
- They lack enough artifact lineage for reproduction.

## The three carrier surfaces

**Tool-call channels** encode information in action choices. The choice can be which fake tool the
agent selects, the order of calls, arguments, timing, or state carried between turns. The default
harness returns canned responses. Its tool names resemble filesystem, shell and web operations, but
the handlers do not do those operations.

**Document and text carriers** encode information in representations that an agent reads or writes.
Examples are Unicode code points, whitespace, capitalization, JSON key order, headings and composed
file content. Most of these baselines run offline. Whether a real model and transport keep a given
representation is a separate measurement.

**Token channels** use the next-token distribution of a language model. The current arithmetic
coder is a finite-message target-point baseline. It is useful to study exact transport and
detection. I do not claim that it keeps the model distribution or that it instantiates a published
indistinguishability proof.

**Detectors** are next to the channels that they examine. The evaluation helpers do these things:

- They enforce group-disjoint splits.
- They reject contradictory duplicate observations.
- They keep split manifests.
- They report uncertainty and do not turn an arbitrary cutoff into a scientific verdict.

## Evidence boundary

I keep historical reports because failed and superseded runs are useful when their limits are
visible. They do not all describe the current code.

- [analysis/FINDINGS.md](../analysis/FINDINGS.md) is the historical tool-channel notebook. It has a
  release-status warning.
- [analysis/token_likelihood_results.md](../analysis/token_likelihood_results.md) records early
  token-coder observations and their invalidation.
- [analysis/rerun_2026-05-02_lengthmatch.md](../analysis/rerun_2026-05-02_lengthmatch.md) explains
  why length matching alone did not repair the activation-probe study.
- [analysis/capacity_transfer_results.md](../analysis/capacity_transfer_results.md) and
  [analysis/coupled_probe_results.md](../analysis/coupled_probe_results.md) are kept for
  provenance, not as current measurements.
- The notes under [`studies/`](../studies/) record narrow early tool-channel runs. Each note states
  what evidence is missing and which rerun could qualify its claims.

A fresh result is not ready for a public claim unless it records all of these:

- source and model revisions
- payload classification
- split unit
- seeds
- uncertainty method
- raw-artifact lineage
- an exact regeneration command

A negative result also needs a positive control and a detection limit. Without those, the honest
wording is "not yet made to work."

## Safety boundary

Local Ollama on literal loopback is the default model endpoint. The shared client does not inherit
proxy settings, follow redirects or retry a request automatically.

Remote OpenRouter calls require an explicit, unexpired `RemoteEgressPolicy`. The policy is bound to
the exact provider, destination, model action and synthetic-payload confirmation. An API key alone
is not authorization.

Use only synthetic payloads and infrastructure that you own or are authorized to use. Do not put
credentials, personal data or private prompts in `SECRET`, `KEY`, model messages, filenames, logs or
reports. Raw experiment output belongs in the ignored `findings/` directory until someone reviews
and redacts it.

Read [SECURITY.md](../SECURITY.md) for the threat boundary and the private reporting path.

## Repository map

```text
src/
  agent.py                 local agent loop
  harness.py               canned tool implementations and trace capture
  client.py                loopback client and policy-bound remote client
  remote_policy.py         explicit remote-egress capability
  channels/                tool-call and text-carrier implementations
  token_stego/             token coder, transport, probes and detectors
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

CI also audits the complete exported lock graph for known advisories. It also scans the Git history
for credential signatures. These checks have detection limits. An advisory scanner knows only its
current database. A secret scanner cannot recognize arbitrary operational metadata. Review
filenames, reports and history as well.

## Related work

- [PNR-KE: Undetectable Conversations Between AI Agents](https://arxiv.org/abs/2604.04757)
- [Efficient Provably Secure Linguistic Steganography via Range Coding](https://arxiv.org/abs/2604.08052)
- [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion](https://arxiv.org/abs/2410.03768)
- [Perfectly Secure Steganography Using Minimum Entropy Coupling](https://arxiv.org/abs/2210.14889)
