from typing import Any, Callable

import httpx
import pytest

from thinai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    ChatChunk,
    Message,
    NotFoundError,
    StreamError,
    Thinai,
)

from .conftest import BASE, Recorder, ndjson

PS = httpx.Response(
    200, json={"models": [{"name": "lfm2.5-350m-q8_0", "model": "lfm2.5-350m-q8_0", "size": 0}]}
)
TAGS = {
    "models": [
        {
            "name": "lfm2.5-350m-q8_0",
            "model": "lfm2.5-350m-q8_0",
            "modified_at": "2026-09-14T08:42:47.850081Z",
            "size": 379217632,
            "digest": "",
            "details": {"format": "gguf", "family": "llama"},
        }
    ]
}
CHAT_DONE = {
    "model": "lfm2.5-350m-q8_0",
    "created_at": "2026-09-14T09:00:00Z",
    "message": {"role": "assistant", "content": "Hello!"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 3_000_000_000,
    "load_duration": 0,
    "prompt_eval_count": 12,
    "prompt_eval_duration": 500_000_000,
    "eval_count": 20,
    "eval_duration": 2_000_000_000,
}

MakeClient = Callable[..., Thinai]


def test_env_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINAI_HOST", "10.0.0.7:9000")
    assert Thinai().base_url == "http://10.0.0.7:9000"


def test_no_host_uses_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    from thinai import Server

    monkeypatch.setattr(
        "thinai.client.find_server",
        lambda **kw: Server(host="10.0.0.9", port=kw["port"], base_url="http://10.0.0.9:11434"),
    )
    assert Thinai().base_url == "http://10.0.0.9:11434"


def test_is_thinai(make_client: MakeClient) -> None:
    ok = make_client(
        Recorder({"/": httpx.Response(200, text="Thinai is running. See /api/tags or /v1/models.")})
    )
    ollama = make_client(Recorder({"/": httpx.Response(200, text="Ollama is running")}))
    assert ok.is_thinai() is True
    assert ollama.is_thinai() is False


def test_models_and_running(make_client: MakeClient) -> None:
    client = make_client(Recorder({"/api/tags": httpx.Response(200, json=TAGS), "/api/ps": PS}))
    [model] = client.models()
    assert model.name == "lfm2.5-350m-q8_0"
    assert model.size == 379217632
    assert model.format == "gguf"
    assert [m.name for m in client.running()] == ["lfm2.5-350m-q8_0"]


def test_show_parses_context_cap(make_client: MakeClient) -> None:
    show = {
        "details": {"family": "lfm2"},
        "model_info": {
            "size": 379217632,
            "general.architecture": "lfm2",
            "lfm2.context_length": 128000,
            "lfm2.block_count": 16,
            "thinai.context_cap": 32768,
        },
    }
    recorder = Recorder({"/api/show": httpx.Response(200, json=show), "/api/ps": PS})
    info = make_client(recorder).show()
    assert recorder.last_body("/api/show") == {"name": "lfm2.5-350m-q8_0"}
    assert (info.architecture, info.context_length, info.context_cap, info.block_count) == (
        "lfm2",
        128000,
        32768,
        16,
    )
    assert info.embedding_length is None


def test_chat_payload_uses_loaded_model(make_client: MakeClient) -> None:
    recorder = Recorder({"/api/ps": PS, "/api/chat": httpx.Response(200, json=CHAT_DONE)})
    response = make_client(recorder).chat(
        [Message.system("be brief"), {"role": "user", "content": "hi"}],
        temperature=0.2,
        num_ctx=2048,
        stop=["\n\n"],
    )
    assert recorder.last_body("/api/chat") == {
        "model": "lfm2.5-350m-q8_0",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 2048, "stop": ["\n\n"]},
    }
    assert response.content == "Hello!"
    assert response.done_reason == "stop"
    assert response.metrics is not None
    assert response.metrics.tokens_per_second == pytest.approx(10.0)
    assert response.metrics.prompt_tokens_per_second == pytest.approx(24.0)


def test_explicit_model_skips_ps(make_client: MakeClient) -> None:
    recorder = Recorder({"/api/chat": httpx.Response(200, json=CHAT_DONE)})
    make_client(recorder, model="gemma-3-270m-it-q8_0").chat("hi")
    assert recorder.paths() == ["/api/chat"]
    assert recorder.last_body("/api/chat")["model"] == "gemma-3-270m-it-q8_0"


def test_chat_stream(make_client: MakeClient) -> None:
    chunk = {
        "model": "m",
        "created_at": "t",
        "message": {"role": "assistant", "content": "Hel"},
        "done": False,
    }
    chunk2 = {**chunk, "message": {"role": "assistant", "content": "lo"}}
    final = {**CHAT_DONE, "message": {"role": "assistant", "content": ""}}
    recorder = Recorder({"/api/chat": ndjson(chunk, chunk2, final)})
    chunks = list(make_client(recorder, model="m").chat("hi", stream=True))
    assert recorder.last_body("/api/chat")["stream"] is True
    assert all(isinstance(c, ChatChunk) for c in chunks)
    assert "".join(c.content for c in chunks) == "Hello"
    assert [c.done for c in chunks] == [False, False, True]
    assert chunks[0].metrics is None
    assert chunks[-1].metrics is not None and chunks[-1].metrics.eval_count == 20


def test_chat_stream_error_line(make_client: MakeClient) -> None:
    chunk = {"model": "m", "message": {"role": "assistant", "content": "a"}, "done": False}
    recorder = Recorder({"/api/chat": ndjson(chunk, {"error": "engine crashed", "done": True})})
    stream = make_client(recorder, model="m").chat("hi", stream=True)
    assert next(stream).content == "a"
    with pytest.raises(StreamError, match="engine crashed"):
        next(stream)


def test_chat_stream_http_error(make_client: MakeClient) -> None:
    recorder = Recorder({"/api/chat": httpx.Response(404, json={"error": "model not found: nope"})})
    with pytest.raises(NotFoundError, match="model not found"):
        list(make_client(recorder).chat("hi", model="nope", stream=True))


def test_chat_tool_calls(make_client: MakeClient) -> None:
    done = {
        **CHAT_DONE,
        "done_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Chennai"}}}],
        },
    }
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    recorder = Recorder({"/api/chat": httpx.Response(200, json=done)})
    response = make_client(recorder, model="m").chat("weather?", tools=tools)
    assert recorder.last_body("/api/chat")["tools"] == tools
    [call] = response.tool_calls
    assert (call.name, call.arguments) == ("get_weather", {"city": "Chennai"})


def test_generate(make_client: MakeClient) -> None:
    done = {
        "model": "m",
        "response": "1, 2, 3",
        "done": True,
        "done_reason": "length",
        "eval_count": 8,
    }
    recorder = Recorder({"/api/generate": httpx.Response(200, json=done)})
    response = make_client(recorder, model="m").generate("count", num_predict=8)
    assert recorder.last_body("/api/generate") == {
        "model": "m",
        "prompt": "count",
        "stream": False,
        "options": {"num_predict": 8},
    }
    assert (response.response, response.done_reason) == ("1, 2, 3", "length")


def test_generate_stream(make_client: MakeClient) -> None:
    recorder = Recorder(
        {
            "/api/generate": ndjson(
                {"model": "m", "response": "a", "done": False},
                {"model": "m", "response": "b", "done": True, "done_reason": "stop"},
            )
        }
    )
    chunks = list(make_client(recorder, model="m").generate("x", stream=True))
    assert "".join(c.response for c in chunks) == "ab"


def test_embed(make_client: MakeClient) -> None:
    payload = {"model": "e", "embeddings": [[0.1, 0.2], [0.3, 0.4]], "prompt_eval_count": 4}
    recorder = Recorder({"/api/embed": httpx.Response(200, json=payload)})
    response = make_client(recorder).embed(("hello", "world"), model="e")
    assert recorder.last_body("/api/embed") == {"model": "e", "input": ["hello", "world"]}
    assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_bad_request(make_client: MakeClient) -> None:
    recorder = Recorder(
        {"/api/embed": httpx.Response(400, json={"error": "model has no pooling layer"})}
    )
    with pytest.raises(BadRequestError) as info:
        make_client(recorder).embed("x", model="chat")
    assert info.value.status_code == 400


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("refused"), APIConnectionError),
        (httpx.ReadTimeout("slow"), APITimeoutError),
    ],
)
def test_transport_errors(make_client: MakeClient, exc: Exception, expected: type) -> None:
    def handler(request: httpx.Request) -> Any:
        raise exc

    with pytest.raises(expected, match="192.168.1.36"):
        make_client(handler).models()


def test_api_key_header(make_client: MakeClient) -> None:
    recorder = Recorder({"/api/tags": httpx.Response(200, json={"models": []})})
    make_client(recorder, api_key="secret").models()
    assert recorder.requests[0].headers["authorization"] == "Bearer secret"


def test_openai_helper(make_client: MakeClient) -> None:
    pytest.importorskip("openai")
    client = make_client(Recorder({})).openai()
    assert str(client.base_url) == f"{BASE}/v1/"


def test_chat_images_top_k_think(make_client: MakeClient) -> None:
    recorder = Recorder({"/api/chat": httpx.Response(200, json=CHAT_DONE)})
    png = b"\x89PNG\r\n\x1a\n"
    make_client(recorder, model="m").chat(
        "describe", images=[png], top_k=40, think=False, temperature=0.1
    )
    sent = recorder.last_body("/api/chat")
    assert sent["think"] is False
    assert sent["options"] == {"temperature": 0.1, "top_k": 40}
    content = sent["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_generate_think_omitted_when_unset(make_client: MakeClient) -> None:
    done = {"model": "m", "response": "x", "done": True}
    recorder = Recorder({"/api/generate": httpx.Response(200, json=done)})
    make_client(recorder, model="m").generate("p", think=True, top_k=5)
    assert recorder.last_body("/api/generate")["think"] is True
    make_client(recorder, model="m").generate("p")
    assert "think" not in recorder.last_body("/api/generate")


def test_health(make_client: MakeClient) -> None:
    recorder = Recorder({"/healthz": httpx.Response(200, json={"status": "ok"})})
    client = make_client(recorder, api_key="secret")
    assert client.health() is True
    assert "authorization" not in recorder.requests[0].headers
    assert make_client(Recorder({})).health() is False

    def down(request: httpx.Request) -> Any:
        raise httpx.ConnectError("refused")

    assert make_client(down).health() is False


def test_is_thinai_with_token_challenge(make_client: MakeClient) -> None:
    challenge = httpx.Response(
        401,
        json={"error": "Missing or wrong API key."},
        headers={"www-authenticate": 'Bearer realm="thinai"'},
    )
    assert make_client(Recorder({"/": challenge})).is_thinai() is True
    ok = httpx.Response(200, text="Thinai is running. See /api/tags or /v1/models.")
    assert make_client(Recorder({"/": ok})).is_thinai() is True
    assert make_client(Recorder({"/": httpx.Response(200, text="Ollama")})).is_thinai() is False


def test_missing_token_raises_auth_error(make_client: MakeClient) -> None:
    from thinai import AuthenticationError

    unauthorized = httpx.Response(
        401,
        json={"error": 'Missing or wrong API key. Send "Authorization: Bearer <token>".'},
        headers={"www-authenticate": 'Bearer realm="thinai"'},
    )
    with pytest.raises(AuthenticationError, match="THINAI_API_KEY") as info:
        make_client(Recorder({"/api/tags": unauthorized})).models()
    assert info.value.status_code == 401


def test_public_url_client() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"models": []})

    client = Thinai(
        "https://abc-def.trycloudflare.com",
        api_key="tok",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert client.base_url == "https://abc-def.trycloudflare.com"
    client.models()
    assert str(seen[0].url) == "https://abc-def.trycloudflare.com/api/tags"
    assert seen[0].headers["authorization"] == "Bearer tok"


def test_api_key_from_env(make_client: MakeClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THINAI_API_KEY", "envtok")
    recorder = Recorder({"/api/tags": httpx.Response(200, json={"models": []})})
    make_client(recorder).models()
    assert recorder.requests[0].headers["authorization"] == "Bearer envtok"
