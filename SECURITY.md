# Security policy

## Research boundary

This repository studies covert communication in controlled AI-agent
experiments. Use synthetic payloads, fake tools, public models and data, and
infrastructure you own or are authorized to use. Real credentials as payloads,
private data, live-target exfiltration, persistence, and deployment-oriented
bypass work are out of scope.

## What stays local

The fake tool harness returns canned responses. Its `read_file`, `write_file`,
`search_web`, and `run_command` names do not touch a filesystem, shell, or web
service. That guarantee applies to those handlers, not to every experiment in
the repository.

The core client defaults to Ollama on literal loopback. Its HTTP transport does
not inherit proxy variables or follow redirects, and the SDK cannot retry a
request automatically.

## What can use a network

Some experiments load model libraries or call an OpenAI-compatible provider.
Package installation and model acquisition can also contact external services.
Read an experiment before running it and use only synthetic prompts.

Remote inference through `src.client` fails closed unless the caller supplies a
live `RemoteEgressPolicy`. The policy binds one approval record, provider,
HTTPS destination, action, expiry, and synthetic-payload confirmation. An API
key or environment variable cannot create that capability by itself. The model,
destination, transport, and policy are checked again before each request.

This in-process check prevents accidental egress. It is not a sandbox: Python
code that already has arbitrary execution can bypass library policy, so untrusted
experiments still need a network namespace, container policy, or host firewall.

## Secrets and artifacts

`.env` and `.local/` are ignored. Keep API keys and SSH details there, set
private file permissions, and never put credentials or personal data in prompts,
logs, filenames, traces, or reports. Raw findings may contain model output; keep
their directory owner-only until you have reviewed and redacted it.

Environment variables named `SECRET`, `KEY`, or `PAYLOAD` are experiment inputs,
not secret stores. Give them fixed synthetic values only. Some historical drivers
write their inputs and decoded outputs to ignored findings, so a real credential
can leak into backups or terminal history even when Git ignores the file.

The repository's demonstrations use fixed harmless payloads. New evidence
should record its payload classification, destinations, retention, and exact
regeneration path before execution.

## Reporting

Use GitHub's private vulnerability-reporting channel when it is available. Do
not paste credentials, private prompts, or working exploit details into a public
issue. If a technique affects a production framework, contact that vendor before
publishing the details.
