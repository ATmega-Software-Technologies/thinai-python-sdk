"""Python SDK for Thinai: LLMs running on an Android phone, served over your Wi-Fi.

>>> from thinai import Thinai
>>> client = Thinai()            # finds the phone on the local network
>>> print(client.chat("Hello!").content)
"""

from ._base import DEFAULT_PORT
from ._version import __version__
from .async_client import AsyncThinai
from .client import Thinai
from .discovery import adiscover, afind_server, discover, find_server
from .errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    DiscoveryError,
    NotFoundError,
    ServerError,
    StreamError,
    ThinaiError,
)
from .types import (
    ChatChunk,
    ChatResponse,
    EmbedResponse,
    GenerateChunk,
    GenerateResponse,
    Message,
    Metrics,
    Model,
    ModelInfo,
    RunningModel,
    Server,
    ToolCall,
)

__all__ = [
    "DEFAULT_PORT",
    "__version__",
    "Thinai",
    "AsyncThinai",
    "discover",
    "adiscover",
    "find_server",
    "afind_server",
    "ThinaiError",
    "DiscoveryError",
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "BadRequestError",
    "AuthenticationError",
    "NotFoundError",
    "ServerError",
    "StreamError",
    "ChatChunk",
    "ChatResponse",
    "EmbedResponse",
    "GenerateChunk",
    "GenerateResponse",
    "Message",
    "Metrics",
    "Model",
    "ModelInfo",
    "RunningModel",
    "Server",
    "ToolCall",
]
