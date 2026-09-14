"""Exceptions raised by the Thinai SDK."""

from __future__ import annotations

from typing import Optional


class ThinaiError(Exception):
    """Base class for every error raised by this package."""


class DiscoveryError(ThinaiError):
    """No Thinai server could be found on the local network."""


class APIConnectionError(ThinaiError):
    """The phone could not be reached (wrong IP, server stopped, different Wi-Fi)."""


class APITimeoutError(APIConnectionError):
    """The request timed out."""


class StreamError(ThinaiError):
    """The server reported an error in the middle of a streamed response."""


class APIStatusError(ThinaiError):
    """The server answered with an HTTP error status."""

    def __init__(self, message: str, *, status_code: int, body: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        return f"{self.status_code}: {self.message}"


class BadRequestError(APIStatusError):
    """HTTP 400: the request was malformed, or the model can't do what was asked."""


class AuthenticationError(APIStatusError):
    """HTTP 401: the server requires a bearer token."""


class NotFoundError(APIStatusError):
    """HTTP 404: unknown model or route."""


class ServerError(APIStatusError):
    """HTTP 5xx: inference failed on the phone."""
