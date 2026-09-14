"""Blocking client for the Thinai API."""

from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
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
from .discovery import DEFAULT_PROBE_TIMEOUT, find_server
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
    from openai import OpenAI

Stop = Union[str, Sequence[str]]


class Thinai:
    """Client for a Thinai app serving models on the local network.

    >>> client = Thinai()                      # find the phone automatically
    >>> client = Thinai("192.168.1.36")        # or connect to a known IP
    >>> client.chat("Hello!").content

    Args:
        host: IP, ``ip:port`` or URL of the phone. Falls back to ``$THINAI_HOST``,
            then to scanning the local network.
        port: Server port shown in the app (default 11434).
        model: Default model id. When unset, requests use the model loaded on the phone.
        api_key: Bearer token, if the server requires one (``$THINAI_API_KEY``).
        timeout: httpx timeout. Default: 5s to connect, no read limit.
        discover_timeout: Per-host probe timeout when scanning the network.
        headers: Extra headers sent with every request.
        http_client: Bring your own ``httpx.Client`` (it won't be closed for you).
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
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        host = host or os.environ.get(_base.ENV_HOST)
        if host:
            self.base_url = _base.normalize_base_url(host, port)
        else:
            self.base_url = find_server(port=port, timeout=discover_timeout).base_url
        self.model = model
        self._api_key = api_key or os.environ.get(_base.ENV_API_KEY)
        self._headers = _base.build_headers(self._api_key, headers)
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout, trust_env=False)

    # -- lifecycle -----------------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Thinai:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Thinai(base_url={self.base_url!r}, model={self.model!r})"

    # -- transport -----------------------------------------------------------------------------

    def _request(
        self, method: str, path: str, json: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        try:
            response = self._http.request(
                method, self.base_url + path, json=json, headers=self._headers
            )
        except httpx.TransportError as exc:
            raise _base.transport_error(exc, self.base_url) from exc
        _base.raise_for_status(response)
        return response

    def _stream(self, path: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        try:
            with self._http.stream(
                "POST", self.base_url + path, json=payload, headers=self._headers
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    _base.raise_for_status(response)
                for line in response.iter_lines():
                    data = _base.parse_ndjson_line(line)
                    if data is not None:
                        yield data
        except httpx.TransportError as exc:
            raise _base.transport_error(exc, self.base_url) from exc

    def _resolve_model(self, model: Optional[str]) -> Optional[str]:
        if model or self.model:
            return model or self.model
        # The server would otherwise pick the first installed model alphabetically,
        # which may force the phone to swap models. Prefer the one already loaded.
        running = self.running()
        return running[0].name if running else None

    # -- models --------------------------------------------------------------------------------

    def is_thinai(self) -> bool:
        """True if ``base_url`` answers like a running Thinai app."""
        try:
            response = self._http.get(self.base_url + "/", headers=self._headers, timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200 and response.text.startswith(FINGERPRINT)

    def models(self) -> List[Model]:
        """Models installed on the phone."""
        data = self._request("GET", "/api/tags").json()
        return [Model.from_dict(m) for m in data.get("models", [])]

    def running(self) -> List[RunningModel]:
        """The model currently loaded (empty if none)."""
        data = self._request("GET", "/api/ps").json()
        return [RunningModel.from_dict(m) for m in data.get("models", [])]

    def show(self, model: Optional[str] = None) -> ModelInfo:
        """Architecture and context limits for a model (default: the loaded one)."""
        name = self._resolve_model(model)
        if not name:
            raise ValueError("no model given and no model is loaded on the phone")
        data = self._request("POST", "/api/show", json={"name": name}).json()
        return ModelInfo.from_dict(name, data)

    # -- chat ----------------------------------------------------------------------------------

    @overload
    def chat(
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
    def chat(
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
    ) -> Iterator[ChatChunk]: ...

    def chat(
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
    ) -> Union[ChatResponse, Iterator[ChatChunk]]:
        """Chat with a model on the phone (``/api/chat``).

        Args:
            messages: A prompt string, one message, or a list of messages
                (dicts or :class:`thinai.Message`).
            model: Model id. Default: the client's ``model``, else the loaded model.
            stream: Yield :class:`ChatChunk` objects as tokens arrive.
            tools: Tool definitions in OpenAI format.
            options: Raw Ollama-style options; keyword arguments below override it.
            temperature: 0.0-2.0.
            num_ctx: Context window; clamped by the phone to ``show().context_cap``.
            top_p: Nucleus sampling.
            num_predict: Maximum tokens to generate.
            stop: Stop sequence or up to 4 sequences.
            num_gpu: ``0`` forces CPU inference.
        """
        payload = _base.compact(
            {
                "model": self._resolve_model(model),
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
            return (ChatChunk.from_dict(d) for d in self._stream("/api/chat", payload))
        return ChatResponse.from_dict(self._request("POST", "/api/chat", json=payload).json())

    # -- generate ------------------------------------------------------------------------------

    @overload
    def generate(
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
    def generate(
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
    ) -> Iterator[GenerateChunk]: ...

    def generate(
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
    ) -> Union[GenerateResponse, Iterator[GenerateChunk]]:
        """Complete a single prompt (``/api/generate``). Options match :meth:`chat`."""
        payload = _base.compact(
            {
                "model": self._resolve_model(model),
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
            return (GenerateChunk.from_dict(d) for d in self._stream("/api/generate", payload))
        return GenerateResponse.from_dict(
            self._request("POST", "/api/generate", json=payload).json()
        )

    # -- embeddings ----------------------------------------------------------------------------

    def embed(
        self,
        input: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        options: Optional[Mapping[str, Any]] = None,
        num_ctx: Optional[int] = None,
        num_gpu: Optional[int] = None,
    ) -> EmbedResponse:
        """Embed one or more strings (``/api/embed``). Needs an embedding model on the phone."""
        payload = _base.compact(
            {
                "model": model or self.model,
                "input": input if isinstance(input, str) else list(input),
                "options": _base.build_options(options, num_ctx=num_ctx, num_gpu=num_gpu),
            }
        )
        return EmbedResponse.from_dict(self._request("POST", "/api/embed", json=payload).json())

    # -- OpenAI compatibility ------------------------------------------------------------------

    def openai(self, **kwargs: Any) -> OpenAI:
        """An ``openai.OpenAI`` client pointed at this phone's ``/v1`` API.

        Requires ``pip install 'thinai[openai]'``.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _base.missing_openai() from exc
        kwargs.setdefault("default_headers", self._headers or None)
        return OpenAI(base_url=self.base_url + "/v1", api_key=self._api_key or "thinai", **kwargs)
