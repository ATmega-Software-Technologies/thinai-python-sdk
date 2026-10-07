# Security Policy

## Supported versions

The project is pre-1.0, so security fixes only go into the latest minor release.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a vulnerability

**Please don't report security issues through public GitHub issues, pull requests, or discussions.**

Report privately through one of these:

1. **GitHub private vulnerability reporting (preferred):** [open a draft advisory](https://github.com/ATmega-Software-Technologies/thinai-python-sdk/security/advisories/new).
2. **Email:** dev.atmega@gmail.com. Put `[thinai-sdk security]` in the subject line.

Please include:

- the affected `thinai` version, Python version, and OS
- steps to reproduce or a minimal proof of concept
- the impact: what an attacker can do, and from where (same machine, same Wi-Fi, or remote)
- a suggested fix, if you have one

## What to expect

- **Acknowledgement** within 3 business days.
- **Initial assessment** within 7 days. We'll confirm whether it's in scope and how severe it is.
- **Fix or mitigation** within 90 days of the report, usually sooner. We'll keep you updated on progress.
- **Disclosure** is coordinated with you through a GitHub Security Advisory, and we'll request a CVE where it applies. Please hold off on publishing details until a fix is released or the 90 days are up.
- **Credit:** reporters are named in the advisory unless they'd rather not be.

## Scope

### In scope

The `thinai` Python package in `src/`, the `thinai` command line, and this repository's CI and release workflows. For example:

- Crafted server responses (JSON, NDJSON streams, or error bodies) that cause code execution, path traversal, unbounded memory or CPU use, or crash the host process in an unexpected way.
- The API key (`api_key` / `$THINAI_API_KEY`) leaking to hosts it wasn't meant for, such as addresses probed during discovery, redirects, logs, or exception messages.
- `client.openai()` forwarding credentials or configuration it shouldn't.
- The CLI writing to or reading from unexpected locations on disk.
- Supply-chain issues in the build or publish pipeline (`.github/workflows/`).

### Out of scope

These are known and documented in the README's "Good to know" section, or they belong somewhere else:

- **No authentication or TLS between the SDK and the phone.** The Thinai server speaks plain HTTP on the LAN by design. Anyone on the same network can reach it and read the traffic.
- **LAN discovery.** `Thinai()` and `discover()` probe the local subnet on purpose. A device on the same Wi-Fi that answers with the Thinai fingerprint will be treated as a Thinai server. Pass an explicit `host` (or set `$THINAI_HOST`) on untrusted networks.
- Model output: hallucinations, jailbreaks, prompt injection in content the model returns.
- Vulnerabilities in dependencies (`httpx`, `openai`), unless the way this SDK uses them makes the problem worse. Please report those upstream.
- **The Thinai Android app** is a separate product. You can send app issues to the same email address, but this repository's advisories don't cover them.

## For contributors

This project accepts outside contributions. To keep it secure:

- Never commit secrets, tokens, or real device IPs from private networks you don't control.
- Keep new dependencies to a minimum and justify each one in the PR. Use version ranges in `pyproject.toml` and commit the updated `uv.lock`.
- Never `eval`/`exec` or unpickle server responses. Treat everything that comes back from the network as untrusted input.
- Don't send the API key or other headers to hosts the user didn't choose (for example, during discovery scans).
- If a change affects networking, discovery, credentials, or the CLI's file handling, say so in the PR description so it gets a closer review.
- If you find a vulnerability while working on a contribution, report it privately as described above. Don't fix it in a public PR first.
