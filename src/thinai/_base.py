"""Transport-independent helpers shared by the sync and async clients."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from urllib.parse import urlsplit

import httpx

from .errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    ServerError,
    StreamError,
)
from .types import Message

DEFAULT_PORT = 11434
FINGERPRINT = "Thinai"
"""``GET /`` on the app answers ``Thinai is running. ...`` (Ollama says ``Ollama is running``)."""

DEFAULT_TIMEOUT = httpx.Timeout(None, connect=5.0)
"""No read timeout: phone inference is slow and the server runs one request at a time."""

ENV_HOST = "THINAI_HOST"
ENV_API_KEY = "THINAI_API_KEY"

MessageInput = Union[Message, Mapping[str, Any]]
MessagesInput = Union[str, MessageInput, Sequence[MessageInput]]


def normalize_base_url(host: str, port: int = DEFAULT_PORT) -> str:
    """Accept ``192.168.1.36``, ``192.168.1.36:8080`` or ``http://phone.local:11434/``."""
    raw = host.strip()
    if not raw:
        raise ValueError("host must not be empty")
    if "://" not in raw:
        raw = f"http://{raw}"
    parts = urlsplit(raw)
    if not parts.hostname:
        raise ValueError(f"invalid host: {host!r}")
    hostname = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{hostname}:{parts.port or port}{parts.path.rstrip('/')}"


def build_headers(
    api_key: Optional[str], headers: Optional[Mapping[str, str]]
) -> Dict[str, str]:
    out = dict(headers or {})
    if api_key:
        out.setdefault("Authorization", f"Bearer {api_key}")
    return out


def build_options(options: Optional[Mapping[str, Any]], **values: Any) -> Optional[Dict[str, Any]]:
    """Merge explicit keyword options over the ``options`` dict, dropping unset ones."""
    merged = dict(options or {})
    merged.update({k: v for k, v in values.items() if v is not None})
    return merged or None


def normalize_messages(messages: MessagesInput) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, (Message, Mapping)):
        messages = [messages]
    out: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Message):
            out.append(message.to_dict())
        elif isinstance(message, Mapping):
            out.append(dict(message))
        else:
            raise TypeError(
                f"messages must be dicts or thinai.Message objects, got {type(message).__name__}"
            )
    return out


def compact(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def raise_for_status(response: httpx.Response) -> None:
    code = response.status_code
    if code < 400:
        return
    exc_type = {400: BadRequestError, 401: AuthenticationError, 404: NotFoundError}.get(code)
    if exc_type is None:
        exc_type = ServerError if code >= 500 else APIStatusError
    raise exc_type(_error_message(response), status_code=code, body=response.text)


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):  # OpenAI-style {"error": {"message": ...}}
        error = error.get("message")
    return str(error) if error else response.text


def parse_ndjson_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except ValueError as exc:
        raise StreamError(f"invalid line in stream: {line[:200]!r}") from exc
    if not isinstance(data, dict):
        raise StreamError(f"unexpected line in stream: {line[:200]!r}")
    if "error" in data:
        error = data["error"]
        if isinstance(error, dict):
            error = error.get("message", error)
        raise StreamError(str(error))
    return data


def transport_error(exc: httpx.TransportError, base_url: str) -> APIConnectionError:
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(f"request to Thinai at {base_url} timed out")
    return APIConnectionError(
        f"could not reach Thinai at {base_url} ({exc.__class__.__name__}: {exc}). "
        "Is the Server running in the app, with 'Share on local network' on?"
    )


def missing_openai() -> ImportError:
    return ImportError("the openai package is not installed; run: pip install 'thinai[openai]'")
