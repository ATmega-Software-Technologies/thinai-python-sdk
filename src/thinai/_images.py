"""Turn images into the ``data:`` URIs the phone accepts.

The app reads images only from OpenAI-style ``image_url`` content parts holding a
``data:image/...;base64,`` URI. It ignores Ollama's ``images`` field and never fetches
remote URLs, so the SDK inlines the bytes itself.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, Union

ImageInput = Union[str, bytes, "os.PathLike[str]"]
"""A file path, raw image bytes, or a ``data:image/...`` URI."""

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _sniff(data: bytes) -> str:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def image_data_uri(image: ImageInput) -> str:
    """Return ``image`` as a ``data:image/...;base64,...`` URI."""
    if isinstance(image, bytes):
        data, mime = image, _sniff(image)
    else:
        path = os.fspath(image)
        if path.startswith("data:"):
            if not path.startswith("data:image/"):
                raise ValueError("data URI must be an image (data:image/...)")
            return path
        if path.startswith(("http://", "https://")):
            raise ValueError(
                "the phone does not fetch remote images; download it and pass the bytes or path"
            )
        with open(path, "rb") as handle:
            data = handle.read()
        guessed = mimetypes.guess_type(path)[0]
        mime = guessed if guessed and guessed.startswith("image/") else _sniff(data)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def image_part(image: ImageInput) -> Dict[str, Any]:
    """An OpenAI ``image_url`` content part for ``image``."""
    return {"type": "image_url", "image_url": {"url": image_data_uri(image)}}
