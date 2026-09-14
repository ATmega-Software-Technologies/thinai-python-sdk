from typing import Callable

import httpx
import pytest

from thinai import AsyncThinai, DiscoveryError, NotFoundError, Server

from .conftest import Recorder, ndjson

MakeAsync = Callable[..., AsyncThinai]


async def test_async_chat(make_async_client: MakeAsync) -> None:
    done = {
        "model": "m",
        "message": {"role": "assistant", "content": "hi"},
        "done": True,
        "done_reason": "stop",
    }
    ps = httpx.Response(200, json={"models": [{"name": "m"}]})
    recorder = Recorder({"/api/ps": ps, "/api/chat": httpx.Response(200, json=done)})
    async with make_async_client(recorder) as client:
        response = await client.chat("hello")
    assert response.content == "hi"
    assert recorder.last_body("/api/chat")["model"] == "m"


async def test_async_chat_stream(make_async_client: MakeAsync) -> None:
    recorder = Recorder(
        {
            "/api/chat": ndjson(
                {"model": "m", "message": {"role": "assistant", "content": "a"}, "done": False},
                {"model": "m", "message": {"role": "assistant", "content": "b"}, "done": False},
                {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True},
            )
        }
    )
    client = make_async_client(recorder, model="m")
    text = "".join([c.content async for c in await client.chat("x", stream=True)])
    assert text == "ab"


async def test_async_errors(make_async_client: MakeAsync) -> None:
    recorder = Recorder(
        {"/api/generate": httpx.Response(404, json={"error": "model not found: x"})}
    )
    client = make_async_client(recorder)
    with pytest.raises(NotFoundError):
        await client.generate("hi", model="x")
    with pytest.raises(NotFoundError):
        async for _ in await client.generate("hi", model="x", stream=True):
            pass


async def test_lazy_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_find(**kwargs: object) -> Server:
        calls.append(kwargs)
        return Server(host="10.0.0.5", port=11434, base_url="http://10.0.0.5:11434")

    monkeypatch.setattr("thinai.async_client.afind_server", fake_find)
    recorder = Recorder({"/api/tags": httpx.Response(200, json={"models": []})})
    client = AsyncThinai(http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)))
    with pytest.raises(DiscoveryError):
        _ = client.base_url
    assert await client.models() == []
    await client.models()
    assert client.base_url == "http://10.0.0.5:11434"
    assert len(calls) == 1
    assert str(recorder.requests[0].url) == "http://10.0.0.5:11434/api/tags"
