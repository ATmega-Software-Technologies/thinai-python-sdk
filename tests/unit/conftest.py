import json
from typing import Any, Callable, Dict, List

import httpx
import pytest

from thinai import AsyncThinai, Thinai

HOST = "192.168.1.36"
BASE = f"http://{HOST}:11434"

Handler = Callable[[httpx.Request], httpx.Response]


def ndjson(*objects: Dict[str, Any]) -> httpx.Response:
    body = "".join(json.dumps(o) + "\n" for o in objects)
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "application/x-ndjson"}
    )


def body(request: httpx.Request) -> Dict[str, Any]:
    return json.loads(request.content) if request.content else {}


class Recorder:
    """Routes requests by path and remembers them."""

    def __init__(self, routes: Dict[str, Any]) -> None:
        self.routes = routes
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, text="Route not found")
        return route(request) if callable(route) else route

    def paths(self) -> List[str]:
        return [r.url.path for r in self.requests]

    def last_body(self, path: str) -> Dict[str, Any]:
        return body([r for r in self.requests if r.url.path == path][-1])


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THINAI_HOST", raising=False)
    monkeypatch.delenv("THINAI_API_KEY", raising=False)


@pytest.fixture
def make_client() -> Callable[..., Thinai]:
    def factory(handler: Handler, **kwargs: Any) -> Thinai:
        return Thinai(
            HOST, http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs
        )

    return factory


@pytest.fixture
def make_async_client() -> Callable[..., AsyncThinai]:
    def factory(handler: Handler, **kwargs: Any) -> AsyncThinai:
        transport = httpx.MockTransport(handler)
        return AsyncThinai(HOST, http_client=httpx.AsyncClient(transport=transport), **kwargs)

    return factory
