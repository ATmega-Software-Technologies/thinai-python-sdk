"""Asyncio client for the Thinai API."""

from __future__ import annotations

import asyncio
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
)

import httpx

from . import _base
from ._base import DEFAULT_PORT, DEFAULT_TIMEOUT, FINGERPRINT, MessagesInput
from .discovery import DEFAULT_PROBE_TIMEOUT, afind_server
from .errors import DiscoveryError
from .types import (
    ChatChunk,
    ChatResponse,
    EmbedResponse,
    GenerateChunk,
    GenerateResponse,
    Model,
    ModelInfo,
    RunningModel,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

Stop = Union[str, Sequence[str]]


class AsyncThinai:
    """Async client for a Thinai app. Arguments match :class:`thinai.Thinai`.

    Without a host, the network scan runs on the first request, or up front with
    ``await client.connect()`` / ``async with AsyncThinai() as client``.

    >>> async with AsyncThinai() as client:
    ...     async for chunk in await client.chat("Hi", stream=True):
    ...         print(chunk.content, end="")
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = DEFAULT_PORT,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = DEFAULT_TIMEOUT,
        discover_timeout: float = DEFAULT_PROBE_TIMEOUT,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        host = host or os.environ.get(_base.ENV_HOST)
        self._base_url = _base.normalize_base_url(host, port) if host else None
        self._port = port
        self._discover_timeout = discover_timeout
        self._connect_lock: Optional[asyncio.Lock] = None
        self.model = model
        self._api_key = api_key or os.environ.get(_base.ENV_API_KEY)
        self._headers = _base.build_headers(self._api_key, headers)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout, trust_env=False)

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise DiscoveryError(
                "server not located yet: await client.connect() or make a request first"
            )
        return self._base_url

    async def connect(self) -> AsyncThinai:
        """Locate the server now (scans the network if no host was given)."""
        if self._base_url is None:
            if self._connect_lock is None:
                self._connect_lock = asyncio.Lock()
            async with self._connect_lock:
                if self._base_url is None:
                    server = await afind_server(port=self._port, timeout=self._discover_timeout)
                    self._base_url = server.base_url
        return self

    # -- lifecycle -----------------------------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncThinai:
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncThinai(base_url={self._base_url!r}, model={self.model!r})"

    # -- transport -----------------------------------------------------------------------------

    async def _request(
        self, method: str, path: str, json: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        await self.connect()
        try:
            response = await self._http.request(
                method, self.base_url + path, json=json, headers=self._headers
            )
        except httpx.TransportError as exc:
            raise _base.transport_error(exc, self.base_url) from exc
        _base.raise_for_status(response)
        return response

    async def _stream(self, path: str, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        await self.connect()
        try:
            async with self._http.stream(
                "POST", self.base_url + path, json=payload, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _base.raise_for_status(response)
                async for line in response.aiter_lines():
                    data = _base.parse_ndjson_line(line)
                    if data is not None:
                        yield data
        except httpx.TransportError as exc:
            raise _base.transport_error(exc, self.base_url) from exc

    async def _chat_chunks(self, payload: Dict[str, Any]) -> AsyncIterator[ChatChunk]:
        async for data in self._stream("/api/chat", payload):
            yield ChatChunk.from_dict(data)

    async def _generate_chunks(self, payload: Dict[str, Any]) -> AsyncIterator[GenerateChunk]:
        async for data in self._stream("/api/generate", payload):
            yield GenerateChunk.from_dict(data)

    async def _resolve_model(self, model: Optional[str]) -> Optional[str]:
        if model or self.model:
            return model or self.model
        running = await self.running()
        return running[0].name if running else None

    # -- models --------------------------------------------------------------------------------

    async def is_thinai(self) -> bool:
        """True if the server answers like a running Thinai app."""
        try:
            await self.connect()
            response = await self._http.get(
                self.base_url + "/", headers=self._headers, timeout=5.0
            )
        except (httpx.HTTPError, DiscoveryError):
            return False
        return response.status_code == 200 and response.text.startswith(FINGERPRINT)

    async def models(self) -> List[Model]:
        data = (await self._request("GET", "/api/tags")).json()
        return [Model.from_dict(m) for m in data.get("models", [])]

    async def running(self) -> List[RunningModel]:
        data = (await self._request("GET", "/api/ps")).json()
        return [RunningModel.from_dict(m) for m in data.get("models", [])]

    async def show(self, model: Optional[str] = None) -> ModelInfo:
        name = await self._resolve_model(model)
        if not name:
            raise ValueError("no model given and no model is loaded on the phone")
        data = (await self._request("POST", "/api/show", json={"name": name})).json()
        return ModelInfo.from_dict(name, data)

    # -- chat ----------------------------------------------------------------------------------

    @overload
    async def chat(
        self,
        messages: MessagesInput,
        *,
        model: Optional[str] = ...,
        stream: Literal[False] = ...,
        tools: Optional[Sequence[Mapping[str, Any]]] = ...,
        options: Optional[Mapping[str, Any]] = ...,
        temperature: Optional[float] = ...,
        num_ctx: Optional[int] = ...,
        top_p: Optional[float] = ...,
        num_predict: Optional[int] = ...,
        stop: Optional[Stop] = ...,
        num_gpu: Optional[int] = ...,
    ) -> ChatResponse: ...

    @overload
    async def chat(
        self,
        messages: MessagesInput,
        *,
        model: Optional[str] = ...,
        stream: Literal[True],
        tools: Optional[Sequence[Mapping[str, Any]]] = ...,
        options: Optional[Mapping[str, Any]] = ...,
        temperature: Optional[float] = ...,
        num_ctx: Optional[int] = ...,
        top_p: Optional[float] = ...,
        num_predict: Optional[int] = ...,
        stop: Optional[Stop] = ...,
        num_gpu: Optional[int] = ...,
    ) -> AsyncIterator[ChatChunk]: ...

    async def chat(
        self,
        messages: MessagesInput,
        *,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        options: Optional[Mapping[str, Any]] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        top_p: Optional[float] = None,
        num_predict: Optional[int] = None,
        stop: Optional[Stop] = None,
        num_gpu: Optional[int] = None,
    ) -> Union[ChatResponse, AsyncIterator[ChatChunk]]:
        """See :meth:`thinai.Thinai.chat`. With ``stream=True``, ``await`` then ``async for``."""
        payload = _base.compact(
            {
                "model": await self._resolve_model(model),
                "messages": _base.normalize_messages(messages),
                "stream": stream,
                "tools": list(tools) if tools else None,
                "options": _base.build_options(
                    options,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    top_p=top_p,
                    num_predict=num_predict,
                    stop=stop if isinstance(stop, (str, type(None))) else list(stop),
                    num_gpu=num_gpu,
                ),
            }
        )
        if stream:
            return self._chat_chunks(payload)
        response = await self._request("POST", "/api/chat", json=payload)
        return ChatResponse.from_dict(response.json())

    # -- generate ------------------------------------------------------------------------------

    @overload
    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = ...,
        stream: Literal[False] = ...,
        options: Optional[Mapping[str, Any]] = ...,
        temperature: Optional[float] = ...,
        num_ctx: Optional[int] = ...,
        top_p: Optional[float] = ...,
        num_predict: Optional[int] = ...,
        stop: Optional[Stop] = ...,
        num_gpu: Optional[int] = ...,
    ) -> GenerateResponse: ...

    @overload
    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = ...,
        stream: Literal[True],
        options: Optional[Mapping[str, Any]] = ...,
        temperature: Optional[float] = ...,
        num_ctx: Optional[int] = ...,
        top_p: Optional[float] = ...,
        num_predict: Optional[int] = ...,
        stop: Optional[Stop] = ...,
        num_gpu: Optional[int] = ...,
    ) -> AsyncIterator[GenerateChunk]: ...

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        stream: bool = False,
        options: Optional[Mapping[str, Any]] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        top_p: Optional[float] = None,
        num_predict: Optional[int] = None,
        stop: Optional[Stop] = None,
        num_gpu: Optional[int] = None,
    ) -> Union[GenerateResponse, AsyncIterator[GenerateChunk]]:
        """See :meth:`thinai.Thinai.generate`."""
        payload = _base.compact(
            {
                "model": await self._resolve_model(model),
                "prompt": prompt,
                "stream": stream,
                "options": _base.build_options(
                    options,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    top_p=top_p,
                    num_predict=num_predict,
                    stop=stop if isinstance(stop, (str, type(None))) else list(stop),
                    num_gpu=num_gpu,
                ),
            }
        )
        if stream:
            return self._generate_chunks(payload)
        response = await self._request("POST", "/api/generate", json=payload)
        return GenerateResponse.from_dict(response.json())

    # -- embeddings ----------------------------------------------------------------------------

    async def embed(
        self,
        input: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        num_ctx: Optional[int] = None,
        num_gpu: Optional[int] = None,
    ) -> EmbedResponse:
        """See :meth:`thinai.Thinai.embed`."""
        payload = _base.compact(
            {
                "model": model or self.model,
                "input": input if isinstance(input, str) else list(input),
                "options": _base.build_options(options, num_ctx=num_ctx, num_gpu=num_gpu),
            }
        )
        response = await self._request("POST", "/api/embed", json=payload)
        return EmbedResponse.from_dict(response.json())

    # -- OpenAI compatibility ------------------------------------------------------------------

    def openai(self, **kwargs: Any) -> AsyncOpenAI:
        """An ``openai.AsyncOpenAI`` client for this phone. Call after :meth:`connect`."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise _base.missing_openai() from exc
        kwargs.setdefault("default_headers", self._headers or None)
        return AsyncOpenAI(
            base_url=self.base_url + "/v1", api_key=self._api_key or "thinai", **kwargs
        )
