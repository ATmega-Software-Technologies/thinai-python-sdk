# thinai

Python SDK for **[Thinai](https://play.google.com/store/apps/details?id=in.atmega.thinai&hl=en_IN)**, the Android app that runs LLMs offline on your phone and serves them to your Wi-Fi.

Point your laptop at the phone and use its models from Python. There are no API keys and no cloud, and your data stays on your network.

```bash
pip install thinai
```

```python
from thinai import Thinai

client = Thinai()  # finds the phone on your Wi-Fi
print(client.chat("Explain RAG in one sentence.").content)
```

## Set up the phone

1. Open Thinai, download a model, and go to the **Server** tab.
2. Press **Start**.
3. Turn on **Share on local network**.
4. Keep the computer on the same Wi-Fi. The app shows the address, for example `http://192.168.1.36:11434`.

## Connecting

```python
from thinai import Thinai, discover

client = Thinai()                           # scan the local /24 for Thinai
client = Thinai("192.168.1.36")             # known IP
client = Thinai("192.168.1.36", port=8080)  # custom port from the app
client = Thinai("http://192.168.1.36:11434", model="gemma-3-270m-it-q8_0")

for server in discover():                   # every phone on the network
    print(server.base_url, server.models)
```

When you don't pass `host`, the client looks in this order:

1. `$THINAI_HOST`
2. a Thinai app on this machine
3. a scan of your local subnet

The app doesn't advertise itself on the network yet. The scan probes `GET /` on each address and matches the `Thinai is running` reply, which usually takes 1–2 seconds. Pass `discover(subnets=["10.0.0.0/24"])` if your network isn't a /24.

## Chat

```python
from thinai import Message

response = client.chat(
    [Message.system("You are concise."), Message.user("What is a GGUF file?")],
    temperature=0.3,
    num_ctx=4096,      # clamped by the phone to client.show().context_cap
    num_predict=256,
)
print(response.content, response.done_reason)
print(f"{response.metrics.tokens_per_second:.1f} tok/s on the phone")
```

To stream the reply as it's generated:

```python
for chunk in client.chat("Write a haiku about the monsoon.", stream=True):
    print(chunk.content, end="", flush=True)
```

If you leave out `model`, the client uses the model already loaded on the phone. Swapping models on a phone is slow.

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

The phone also serves `/v1/chat/completions`, `/v1/embeddings` and `/v1/models`. To use them through the official client:

```bash
pip install 'thinai[openai]'
```

```python
oai = Thinai().openai()
oai.chat.completions.create(model="lfm2.5-350m-q8_0", messages=[{"role": "user", "content": "hi"}])
```

This means LangChain, LlamaIndex and other OpenAI-compatible tools work too. Give them `client.base_url + "/v1"` as the base URL.

## Command line

```bash
thinai discover
thinai models --host 192.168.1.36
thinai chat "What is the capital of Tamil Nadu?"
```

## Errors

| Exception | When |
| --- | --- |
| `DiscoveryError` | No phone found on the network |
| `APIConnectionError` / `APITimeoutError` | Phone unreachable (server stopped, sharing off, different Wi-Fi) |
| `NotFoundError` | Unknown model id |
| `BadRequestError` | Invalid request, such as embedding with a chat model |
| `ServerError` | Inference failed on the phone |
| `StreamError` | Server reported an error mid-stream |

All of them inherit from `thinai.ThinaiError`.

## Good to know

- **No authentication.** Anyone on the same Wi-Fi can use the phone's models while sharing is on. Only enable it on networks you trust.
- **One request at a time.** The phone processes requests in a queue, so concurrent calls wait. Cancelling a request on the client doesn't stop generation on the phone.
- **`/api/chat` and `/api/generate` can return engine errors as text.** For example, "request exceeds the available context size" may arrive as the reply content instead of an HTTP error.
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
