# Changelog

## 0.2.0

Matches the Thinai app 2.6 local API server.

### Added
- Public URL support: `Thinai("https://….trycloudflare.com", api_key=...)`. `https://` URLs without a port now use 443, and a bare `*.trycloudflare.com` host is treated as https.
- `health()` on both clients, backed by the app's `GET /healthz` (no token needed), and a `thinai health` command.
- Image input for vision models: `chat(..., images=[...])` and `Message.user(text, images=[...])` take file paths, bytes or `data:image/...` URIs and send them as `image_url` content parts, the only image form the app reads.
- `top_k` and `think` keyword arguments on `chat()` and `generate()`.
- `discover(api_key=...)` and `Server.auth_required`; `--api-key` on every CLI command (also read from `$THINAI_API_KEY`).

### Fixed
- Discovery and `is_thinai()` missed phones with LAN sharing on: since app 2.6 every LAN request needs a token, so `GET /` answers 401. The app's `WWW-Authenticate: Bearer realm="thinai"` challenge now counts as a match.
- `AuthenticationError` says how to pass the token.

### Changed
- `https://host` without a port no longer gets `:11434` appended. Add the port to the URL if you serve https on another one.
- README: API tokens, Public URL, images, and engine failures (reported as errors since app 2.5) replace the old "no authentication" notes.

## 0.1.0

First release.

- `Thinai` (sync) and `AsyncThinai` (asyncio) clients for the phone's `/api/*` routes: `chat`, `generate`, `embed`, `models`, `running`, `show`.
- NDJSON streaming, tool calls, and generation metrics (tokens per second).
- LAN discovery with `discover()` / `adiscover()`, which scan the local /24 for the Thinai fingerprint. `Thinai()` connects automatically.
- `client.openai()`, which returns an OpenAI SDK client pointed at the phone's `/v1` API.
- `thinai` command line: `discover`, `models`, `chat`.
