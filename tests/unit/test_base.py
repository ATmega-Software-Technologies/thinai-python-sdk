from pathlib import Path

import httpx
import pytest

from thinai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    Message,
    NotFoundError,
    ServerError,
    StreamError,
)
from thinai._base import (
    attach_images,
    build_options,
    is_thinai_reply,
    normalize_base_url,
    normalize_messages,
    parse_ndjson_line,
    raise_for_status,
)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("192.168.1.36", "http://192.168.1.36:11434"),
        (" 192.168.1.36 ", "http://192.168.1.36:11434"),
        ("192.168.1.36:8080", "http://192.168.1.36:8080"),
        ("http://192.168.1.36:11434/", "http://192.168.1.36:11434"),
        ("https://phone.local:8443/prefix/", "https://phone.local:8443/prefix"),
        ("https://abc-def.trycloudflare.com", "https://abc-def.trycloudflare.com"),
        ("https://abc-def.trycloudflare.com/", "https://abc-def.trycloudflare.com"),
        ("abc-def.trycloudflare.com", "https://abc-def.trycloudflare.com"),
        ("HTTPS://Phone.Example/", "https://phone.example"),
        ("http://phone.local", "http://phone.local:11434"),
    ],
)
def test_normalize_base_url(host: str, expected: str) -> None:
    assert normalize_base_url(host) == expected


def test_normalize_base_url_rejects_empty() -> None:
    with pytest.raises(ValueError):
        normalize_base_url("  ")


def test_build_options_merges_and_drops_none() -> None:
    assert build_options(None, temperature=None) is None
    assert build_options({"top_k": 5, "temperature": 1.0}, temperature=0.2, num_ctx=None) == {
        "top_k": 5,
        "temperature": 0.2,
    }


def test_normalize_messages() -> None:
    assert normalize_messages("hi") == [{"role": "user", "content": "hi"}]
    assert normalize_messages(Message.system("be brief")) == [
        {"role": "system", "content": "be brief"}
    ]
    assert normalize_messages([{"role": "user", "content": "a"}, Message.assistant("b")]) == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    with pytest.raises(TypeError):
        normalize_messages(["plain strings are ambiguous"])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("response", "exc_type", "message"),
    [
        (httpx.Response(400, json={"error": "name is required"}), BadRequestError, "name is"),
        (httpx.Response(401, json={"error": "nope"}), AuthenticationError, "THINAI_API_KEY"),
        (
            httpx.Response(
                401, json={"error": {"message": "Missing or wrong API key.", "type": "x"}}
            ),
            AuthenticationError,
            "Missing or wrong API key",
        ),
        (httpx.Response(404, text="Route not found"), NotFoundError, "Route not found"),
        (
            httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}}),
            ServerError,
            "boom",
        ),
        (httpx.Response(409, text=""), APIStatusError, "Conflict"),
    ],
)
def test_raise_for_status(response: httpx.Response, exc_type: type, message: str) -> None:
    with pytest.raises(exc_type) as info:
        raise_for_status(response)
    assert message in str(info.value)
    assert info.value.status_code == response.status_code


def test_parse_ndjson_line() -> None:
    assert parse_ndjson_line("  ") is None
    assert parse_ndjson_line('{"done": false}\n') == {"done": False}
    with pytest.raises(StreamError, match="context"):
        parse_ndjson_line('{"error": "exceeds context", "done": true}')
    with pytest.raises(StreamError):
        parse_ndjson_line("not json")


def test_is_thinai_reply() -> None:
    challenge = {"www-authenticate": 'Bearer realm="thinai"'}
    assert is_thinai_reply(200, {}, b"Thinai is running.")
    assert not is_thinai_reply(200, {}, b"Ollama is running")
    assert is_thinai_reply(401, challenge, b"")
    assert not is_thinai_reply(401, {"www-authenticate": 'Basic realm="router"'}, b"")
    assert not is_thinai_reply(404, challenge, b"Thinai")


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


def test_attach_images_to_last_user_message(tmp_path: Path) -> None:
    path = tmp_path / "cat.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0")
    messages = normalize_messages([Message.system("s"), Message.user("what is this?")])
    out = attach_images(messages, [PNG, str(path), "data:image/webp;base64,AAAA"])
    assert out[0] == {"role": "system", "content": "s"}
    parts = out[1]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    urls = [p["image_url"]["url"] for p in parts[1:]]
    assert urls[0].startswith("data:image/png;base64,")
    assert urls[1].startswith("data:image/jpeg;base64,")
    assert urls[2] == "data:image/webp;base64,AAAA"
    assert attach_images(messages, None) is messages


def test_attach_images_rejects_remote_and_missing_user() -> None:
    with pytest.raises(ValueError, match="remote"):
        attach_images([{"role": "user", "content": "x"}], ["https://example.com/a.png"])
    with pytest.raises(ValueError, match="user message"):
        attach_images([{"role": "system", "content": "x"}], [PNG])


def test_message_images_to_dict() -> None:
    data = Message.user("look", images=[PNG]).to_dict()
    assert data["content"][0] == {"type": "text", "text": "look"}
    assert data["content"][1]["type"] == "image_url"
    assert Message.user("plain").to_dict() == {"role": "user", "content": "plain"}
