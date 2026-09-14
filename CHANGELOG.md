# Changelog

## 0.1.0

First release.

- `Thinai` (sync) and `AsyncThinai` (asyncio) clients for the phone's `/api/*` routes: `chat`, `generate`, `embed`, `models`, `running`, `show`.
- NDJSON streaming, tool calls, and generation metrics (tokens per second).
- LAN discovery with `discover()` / `adiscover()`, which scan the local /24 for the Thinai fingerprint. `Thinai()` connects automatically.
- `client.openai()`, which returns an OpenAI SDK client pointed at the phone's `/v1` API.
- `thinai` command line: `discover`, `models`, `chat`.
