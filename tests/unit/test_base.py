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
    build_options,
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
        ("https://phone.local/prefix/", "https://phone.local:11434/prefix"),
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
        (httpx.Response(401, json={"error": "unauthorized"}), AuthenticationError, "unauthorized"),
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
