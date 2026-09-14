"""Command line: ``thinai discover``, ``thinai models``, ``thinai chat "hi"``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ._base import DEFAULT_PORT
from ._version import __version__
from .client import Thinai
from .discovery import DEFAULT_PROBE_TIMEOUT, discover
from .errors import ThinaiError


def _size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thinai", description="Use LLMs running in the Thinai Android app."
    )
    parser.add_argument("--version", action="version", version=f"thinai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("discover", help="find Thinai apps on the local network")
    scan.add_argument("--port", type=int, default=DEFAULT_PORT)
    scan.add_argument("--timeout", type=float, default=DEFAULT_PROBE_TIMEOUT)
    scan.add_argument(
        "--subnet", action="append", help="CIDR to scan, e.g. 192.168.1.0/24 (repeatable)"
    )

    for name, help_text in (("models", "list installed models"), ("chat", "send one prompt")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--host", help="phone IP or URL (default: $THINAI_HOST, then scan)")
        cmd.add_argument("--port", type=int, default=DEFAULT_PORT)
        if name == "chat":
            cmd.add_argument("prompt", nargs="+")
            cmd.add_argument("--model")
            cmd.add_argument("--system", help="system prompt")
            cmd.add_argument("--temperature", type=float)
            cmd.add_argument("--no-stream", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "discover":
            servers = discover(port=args.port, timeout=args.timeout, subnets=args.subnet)
            if not servers:
                print("No Thinai servers found.", file=sys.stderr)
                return 1
            for server in servers:
                print(f"{server.base_url}  models: {', '.join(server.models) or '-'}")
            return 0

        with Thinai(args.host, args.port) as client:
            if args.command == "models":
                loaded = {m.name for m in client.running()}
                for model in client.models():
                    marker = "*" if model.name in loaded else " "
                    print(f"{marker} {model.name:40} {_size(model.size):>9}")
                return 0

            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": " ".join(args.prompt)})
            if args.no_stream:
                print(client.chat(messages, model=args.model, temperature=args.temperature).content)
            else:
                for chunk in client.chat(
                    messages, model=args.model, temperature=args.temperature, stream=True
                ):
                    print(chunk.content, end="", flush=True)
                print()
            return 0
    except ThinaiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
