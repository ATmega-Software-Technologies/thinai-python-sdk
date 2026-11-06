# thinai

Python SDK for **[Thinai](https://play.google.com/store/apps/details?id=in.atmega.thinai&hl=en_IN)**, the Android app that runs LLMs offline on your phone and serves them to your Wi-Fi.

Point your laptop at the phone and use its models from Python. No cloud inference: the model runs on the phone, and on your Wi-Fi your data stays on your network.

```bash
pip install thinai
```

```python
from thinai import Thinai

client = Thinai(api_key="<token from the Server page>")  # finds the phone on your Wi-Fi
print(client.chat("Explain RAG in one sentence.").content)
```

Works with Thinai app 2.6 or later. Older versions work too, minus `health()` and the Public URL.

## Set up the phone

1. Open Thinai, download a model, and go to the **Server** tab.
2. Press **Start**.
3. Turn on **Share on local network**.
4. Under **API tokens**, create a token for this computer and copy it. Every LAN or Public URL request needs one; deleting the token cuts that client off at once.
5. Keep the computer on the same Wi-Fi. The app shows the address, for example `http://192.168.1.36:11434`.

To reach the phone from outside your Wi-Fi, turn on **Public URL (Beta)** instead. The app opens a free Cloudflare quick tunnel and shows an `https://….trycloudflare.com` address. Requests and replies pass through Cloudflare, and the address changes when you rotate it or restart the tunnel.

## Connecting

```python
from thinai import Thinai, discover

client = Thinai()                           # scan the local /24 for Thinai
client = Thinai("192.168.1.36")             # known IP
client = Thinai("192.168.1.36", port=8080)  # custom port from the app
client = Thinai("http://192.168.1.36:11434", model="gemma-3-270m-it-q8_0")
client = Thinai("https://abc-xyz.trycloudflare.com", api_key=token)  # Public URL

client.health()                             # True if the server is up (GET /healthz)

for server in discover(api_key=token):      # every phone on the network
    print(server.base_url, server.models, server.auth_required)
```

The token goes in `api_key=` or `$THINAI_API_KEY`, and is sent as `Authorization: Bearer <token>`. Apps on the phone itself calling `127.0.0.1` don't need one. A missing or wrong token raises `AuthenticationError`.

`https://` URLs without a port use 443, which is what the Public URL needs; plain hosts and `http://` URLs get port 11434. A bare `*.trycloudflare.com` host is treated as https.

When you don't pass `host`, the client looks in this order:

1. `$THINAI_HOST`
2. a Thinai app on this machine
3. a scan of your local subnet

The app doesn't advertise itself on the network yet. The scan probes `GET /` on each address and matches the `Thinai is running` reply, or the app's `401` with `WWW-Authenticate: Bearer realm="thinai"` when the server needs a token. It usually takes 1–2 seconds. Finding a server doesn't need a token; listing its models does. Pass `discover(subnets=["10.0.0.0/24"])` if your network isn't a /24.

## Chat

```python
from thinai import Message

response = client.chat(
    [Message.system("You are concise."), Message.user("What is a GGUF file?")],
    temperature=0.3,
    top_k=40,
    num_ctx=4096,      # clamped by the phone to client.show().context_cap
    num_predict=256,
    think=False,       # turn a thinking model's reasoning off
)
print(response.content, response.done_reason)
print(f"{response.metrics.tokens_per_second:.1f} tok/s on the phone")
```

To stream the reply as it's generated:

```python
for chunk in client.chat("Write a haiku about the monsoon.", stream=True):
    print(chunk.content, end="", flush=True)
```

If you leave out `model`, the client uses the model already loaded on the phone. Swapping models on a phone is slow. Sampling values you don't pass fall back to the model's settings in the app.

Images work with a vision model whose image encoder is loaded on the phone (for example Gemma 3 4B). Pass file paths, bytes or `data:image/...` URIs; the SDK inlines them, since the phone never fetches remote URLs:

```python
client.chat("What is in this photo?", images=["receipt.jpg"])
client.chat([Message.user("Compare these.", images=[png_bytes, "b.png"])])
```

Without a vision model loaded, the phone answers `BadRequestError`.

Tool calling uses OpenAI-format tool definitions:

```python
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}}]
response = client.chat("Weather in Chennai?", tools=tools)
for call in response.tool_calls:
    print(call.name, call.arguments)
```

## Generate, embed, models

```python
client.generate("Once upon a time", num_predict=50).response
client.embed(["hello", "world"], model="<embedding-model-id>").embeddings  # needs an embedding model
client.models()          # installed models
client.running()         # loaded model
client.show()            # architecture, trained context_length, context_cap
```

## Async

```python
import asyncio
from thinai import AsyncThinai

async def main():
    async with AsyncThinai() as client:
        async for chunk in await client.chat("Hi!", stream=True):
            print(chunk.content, end="")

asyncio.run(main())
```

## OpenAI-compatible API

The phone also serves `/v1/chat/completions` (streaming, tools, `tool_choice`, images, GBNF `grammar`), `/v1/embeddings` (`encoding_format`, `dimensions`) and `/v1/models`. To use them through the official client, which gets the client's token as its API key:

```bash
pip install 'thinai[openai]'
```

```python
oai = Thinai().openai()
oai.chat.completions.create(model="lfm2.5-350m-q8_0", messages=[{"role": "user", "content": "hi"}])
```

This means LangChain, LlamaIndex and other OpenAI-compatible tools work too. Give them `client.base_url + "/v1"` as the base URL and your token as the API key.

## Command line

```bash
thinai discover
thinai health --host https://abc-xyz.trycloudflare.com
thinai models --host 192.168.1.36 --api-key "$TOKEN"
thinai chat "What is the capital of Tamil Nadu?"   # token from $THINAI_API_KEY
```

## Errors

| Exception | When |
| --- | --- |
| `DiscoveryError` | No phone found on the network |
| `APIConnectionError` / `APITimeoutError` | Phone unreachable (server stopped, sharing off, different Wi-Fi, tunnel closed) |
| `AuthenticationError` | Missing, wrong or deleted API token |
| `NotFoundError` | Unknown model id |
| `BadRequestError` | Invalid request, such as embedding with a chat model, or an image without a vision model |
| `ServerError` | Inference failed on the phone |
| `StreamError` | Server reported an error mid-stream |

All of them inherit from `thinai.ThinaiError`.

## Good to know

- **Tokens protect LAN and Public URL access.** Create one token per client in the app and delete it to revoke that client. `GET /healthz` is the only route open without one, and it returns no data. Traffic on the LAN is plain HTTP, so only share on networks you trust.
- **The Public URL goes through Cloudflare.** It is https end to end to Cloudflare's edge, but Cloudflare sees requests and replies. Use it only for prompts you're comfortable with that.
- **Security issues.** Please report them privately. See [SECURITY.md](SECURITY.md).
- **One request at a time.** The phone processes requests in a queue, so concurrent calls wait. Cancelling a request on the client doesn't stop generation on the phone.
- **Engine failures are errors.** Since app 2.5, "request exceeds the available context size" and similar failures come back as `ServerError` (or `StreamError` mid-stream), not as reply text.
- **`num_ctx` only works on `/api/*` routes.** The OpenAI-compatible `/v1/chat/completions` always uses the context size set in the app.

## Development

```bash
uv sync
uv run pytest                                    # unit tests (offline)
THINAI_HOST=192.168.1.36 uv run pytest -m live   # against a real phone
uv run ruff check . && uv run mypy
uv build && uv run twine check dist/*
```

## License

MIT
