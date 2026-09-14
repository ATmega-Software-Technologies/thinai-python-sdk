"""End-to-end tests against a real phone running Thinai.

    uv run pytest -m live                            # auto-discover the phone
    THINAI_HOST=192.168.1.36 uv run pytest -m live   # or target it directly
"""

import os
from typing import Iterator
from urllib.parse import urlsplit

import pytest

from thinai import APIStatusError, AsyncThinai, NotFoundError, Thinai, discover

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client() -> Iterator[Thinai]:
    host = os.environ.get("THINAI_HOST")
    if not host:
        servers = discover()
        if not servers:
            pytest.skip("no Thinai server found on the local network")
        host = servers[0].base_url
    with Thinai(host) as c:
        yield c


@pytest.fixture(scope="module")
def chat_model(client: Thinai) -> str:
    running = client.running()
    return running[0].name if running else client.models()[0].name


def test_discover_finds_phone(client: Thinai) -> None:
    servers = discover()
    expected = urlsplit(client.base_url).hostname
    match = [s for s in servers if s.host == expected]
    assert match, f"{expected} not in {[s.base_url for s in servers]}"
    assert match[0].models


def test_is_thinai(client: Thinai) -> None:
    assert client.is_thinai()


def test_models_and_running(client: Thinai, chat_model: str) -> None:
    names = [m.name for m in client.models()]
    assert chat_model in names
    assert all(m.size > 0 for m in client.models())
    assert client.running(), "expected a model loaded on the phone"


def test_show(client: Thinai, chat_model: str) -> None:
    info = client.show(chat_model)
    assert info.architecture
    assert info.context_cap and info.context_cap >= 512


def test_chat(client: Thinai, chat_model: str) -> None:
    response = client.chat(
        [{"role": "user", "content": "Reply with one word: hello"}],
        model=chat_model,
        num_predict=32,
        temperature=0,
    )
    assert response.done
    assert response.content.strip()
    assert response.done_reason in ("stop", "length")
    assert response.metrics and response.metrics.eval_count


def test_chat_stream(client: Thinai, chat_model: str) -> None:
    chunks = list(client.chat("Name three colors.", model=chat_model, stream=True, num_predict=24))
    assert len(chunks) >= 2
    assert chunks[-1].done and not any(c.done for c in chunks[:-1])
    assert "".join(c.content for c in chunks).strip()


def test_generate_respects_num_predict(client: Thinai, chat_model: str) -> None:
    response = client.generate(
        "Count from 1 to 100, separated by commas.", model=chat_model, num_predict=8
    )
    assert response.done_reason == "length"
    assert response.response.strip()


def test_generate_stream(client: Thinai, chat_model: str) -> None:
    chunks = list(client.generate("Say hi.", model=chat_model, stream=True, num_predict=16))
    assert chunks[-1].done
    assert "".join(c.response for c in chunks).strip()


def test_unknown_model(client: Thinai) -> None:
    with pytest.raises(NotFoundError):
        client.chat("hi", model="does-not-exist")


async def test_async_client(client: Thinai, chat_model: str) -> None:
    async with AsyncThinai(client.base_url, model=chat_model) as aclient:
        assert await aclient.is_thinai()
        response = await aclient.chat("Say ok.", num_predict=8)
        assert response.content.strip()
        text = "".join(
            [c.content async for c in await aclient.chat("Say hi.", stream=True, num_predict=8)]
        )
        assert text.strip()


def test_openai_compat(client: Thinai, chat_model: str) -> None:
    pytest.importorskip("openai")
    completion = client.openai().chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=16,
    )
    assert completion.choices[0].message.content
    assert [m.id for m in client.openai().models.list()]


def test_embed_with_chat_model_is_rejected(client: Thinai, chat_model: str) -> None:
    """No embedding model is installed on the test phone; a chat model must be refused."""
    embedding_like = [m.name for m in client.models() if "embed" in m.name]
    if embedding_like:
        response = client.embed(["hello", "world"], model=embedding_like[0])
        assert len(response.embeddings) == 2 and response.embeddings[0]
        return
    with pytest.raises(APIStatusError):
        client.embed("hello", model=chat_model)
