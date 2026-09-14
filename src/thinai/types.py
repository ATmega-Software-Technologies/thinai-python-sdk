"""Typed objects returned by the Thinai clients.

Every response object keeps the untouched JSON in ``raw`` so fields the SDK
doesn't model yet are still reachable.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar


def _opt_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


@dataclass
class ToolCall:
    """A function call requested by the model."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        function = data.get("function") or {}
        arguments: Any = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return cls(name=str(function.get("name", "")), arguments=arguments, id=data.get("id"))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"function": {"name": self.name, "arguments": self.arguments}}
        if self.id:
            out["id"] = self.id
        return out


@dataclass
class Message:
    """A chat message. Plain dicts are accepted anywhere a Message is."""

    role: str
    content: Optional[str] = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    name: Optional[str] = None
    """For ``role="tool"`` messages: the name of the tool that produced this result."""

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, content: str, name: str) -> Message:
        return cls(role="tool", content=content, name=name)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        return cls(
            role=str(data.get("role") or "assistant"),
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(t) for t in data.get("tool_calls") or []],
            name=data.get("name") or data.get("tool_name"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [t.to_dict() for t in self.tool_calls]
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class Metrics:
    """Timing counters from a finished generation. Durations are nanoseconds."""

    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Generation speed on the phone."""
        if not self.eval_count or not self.eval_duration:
            return None
        return self.eval_count / (self.eval_duration / 1e9)

    @property
    def prompt_tokens_per_second(self) -> Optional[float]:
        """Prompt processing speed on the phone."""
        if not self.prompt_eval_count or not self.prompt_eval_duration:
            return None
        return self.prompt_eval_count / (self.prompt_eval_duration / 1e9)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Optional[Metrics]:
        names = [f.name for f in dataclasses.fields(cls)]
        if not any(name in data for name in names):
            return None
        return cls(**{name: _opt_int(data.get(name)) for name in names})


_ChatT = TypeVar("_ChatT", bound="ChatResponse")
_GenT = TypeVar("_GenT", bound="GenerateResponse")


@dataclass
class ChatResponse:
    """Result of ``chat()``."""

    model: str
    message: Message
    done: bool = True
    done_reason: Optional[str] = None
    """``"stop"``, ``"length"`` (hit ``num_predict``) or ``"tool_calls"``."""
    created_at: Optional[str] = None
    metrics: Optional[Metrics] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def content(self) -> str:
        return self.message.content or ""

    @property
    def tool_calls(self) -> List[ToolCall]:
        return self.message.tool_calls

    @classmethod
    def from_dict(cls: Type[_ChatT], data: Mapping[str, Any]) -> _ChatT:
        message = data.get("message")
        done = bool(data.get("done", False))
        return cls(
            model=str(data.get("model", "")),
            message=Message.from_dict(message) if message else Message(role="assistant"),
            done=done,
            done_reason=data.get("done_reason"),
            created_at=data.get("created_at"),
            metrics=Metrics.from_dict(data) if done else None,
            raw=dict(data),
        )


class ChatChunk(ChatResponse):
    """One streamed piece of a chat reply. ``content`` holds only the new text.

    The final chunk has ``done=True``, plus ``done_reason``, ``metrics`` and any ``tool_calls``.
    """


@dataclass
class GenerateResponse:
    """Result of ``generate()``."""

    model: str
    response: str
    done: bool = True
    done_reason: Optional[str] = None
    created_at: Optional[str] = None
    metrics: Optional[Metrics] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def content(self) -> str:
        return self.response

    @classmethod
    def from_dict(cls: Type[_GenT], data: Mapping[str, Any]) -> _GenT:
        done = bool(data.get("done", False))
        return cls(
            model=str(data.get("model", "")),
            response=str(data.get("response") or ""),
            done=done,
            done_reason=data.get("done_reason"),
            created_at=data.get("created_at"),
            metrics=Metrics.from_dict(data) if done else None,
            raw=dict(data),
        )


class GenerateChunk(GenerateResponse):
    """One streamed piece of a ``generate()`` reply. ``response`` holds only the new text."""


@dataclass
class EmbedResponse:
    """Result of ``embed()``: one vector per input string."""

    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EmbedResponse:
        return cls(
            model=str(data.get("model", "")),
            embeddings=[list(v) for v in data.get("embeddings") or []],
            total_duration=_opt_int(data.get("total_duration")),
            prompt_eval_count=_opt_int(data.get("prompt_eval_count")),
            raw=dict(data),
        )


@dataclass
class Model:
    """A model installed on the phone (``/api/tags``)."""

    name: str
    size: int = 0
    """File size in bytes."""
    modified_at: Optional[str] = None
    format: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Model:
        details = data.get("details") or {}
        return cls(
            name=str(data.get("name") or data.get("model") or ""),
            size=_opt_int(data.get("size")) or 0,
            modified_at=data.get("modified_at"),
            format=details.get("format"),
            raw=dict(data),
        )


@dataclass
class RunningModel:
    """The model currently loaded on the phone (``/api/ps``)."""

    name: str
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunningModel:
        return cls(name=str(data.get("name") or data.get("model") or ""), raw=dict(data))


@dataclass
class ModelInfo:
    """Details about one model (``/api/show``)."""

    name: str
    architecture: Optional[str] = None
    size: Optional[int] = None
    context_length: Optional[int] = None
    """Context length the model was trained with."""
    context_cap: Optional[int] = None
    """Largest ``num_ctx`` this phone accepts for the model (limited by its RAM)."""
    block_count: Optional[int] = None
    embedding_length: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> ModelInfo:
        info = data.get("model_info") or {}
        arch = info.get("general.architecture") or (data.get("details") or {}).get("family")

        def arch_int(suffix: str) -> Optional[int]:
            return _opt_int(info.get(f"{arch}.{suffix}")) if arch else None

        return cls(
            name=name,
            architecture=arch,
            size=_opt_int(info.get("size")),
            context_length=arch_int("context_length"),
            context_cap=_opt_int(info.get("thinai.context_cap")),
            block_count=arch_int("block_count"),
            embedding_length=arch_int("embedding_length"),
            raw=dict(data),
        )


@dataclass
class Server:
    """A Thinai app found on the network."""

    host: str
    port: int
    base_url: str
    models: List[str] = field(default_factory=list)
