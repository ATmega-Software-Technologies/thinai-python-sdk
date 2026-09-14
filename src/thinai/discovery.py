"""Find Thinai apps on the local network.

The app doesn't advertise itself (no mDNS), so discovery probes ``GET /`` on
every address of the local /24 subnet(s) and keeps the hosts whose reply starts
with ``Thinai``.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, List, Optional, Sequence, Tuple, TypeVar

import httpx

from ._base import DEFAULT_PORT, FINGERPRINT, normalize_base_url
from .errors import DiscoveryError
from .types import Server

DEFAULT_PROBE_TIMEOUT = 0.5
DEFAULT_CONCURRENCY = 128
MAX_SCAN_ADDRESSES = 4096

_T = TypeVar("_T")


def local_ipv4_addresses() -> List[str]:
    """Private IPv4 addresses of this machine, primary interface first."""
    found: List[str] = []

    def add(address: str) -> None:
        try:
            ip = ipaddress.IPv4Address(address)
        except ValueError:
            return
        if ip.is_private and not ip.is_loopback and not ip.is_link_local and address not in found:
            found.append(address)

    try:
        # Connecting a UDP socket sends nothing; it only picks the outbound interface.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.254.254.254", 1))
            add(sock.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(str(info[4][0]))
    except OSError:
        pass
    return found


def candidate_hosts(subnets: Optional[Sequence[str]] = None) -> List[str]:
    """Addresses to probe: the given subnets, or the /24 around each local address."""
    if subnets:
        networks = [ipaddress.IPv4Network(s, strict=False) for s in subnets]
    else:
        networks = [ipaddress.IPv4Network(f"{a}/24", strict=False) for a in local_ipv4_addresses()]

    hosts: List[str] = []
    seen = set()
    for network in networks:
        if network.num_addresses > MAX_SCAN_ADDRESSES:
            raise ValueError(
                f"subnet {network} is too large to scan (max {MAX_SCAN_ADDRESSES} addresses)"
            )
        addresses = [network.network_address] if network.num_addresses == 1 else network.hosts()
        for ip in addresses:
            if str(ip) not in seen:
                seen.add(str(ip))
                hosts.append(str(ip))
    return hosts


async def _probe(
    client: httpx.AsyncClient,
    host: str,
    port: int,
    semaphore: asyncio.Semaphore,
    with_models: bool,
    timeout: float,
) -> Optional[Server]:
    base_url = normalize_base_url(host, port)
    fingerprint = FINGERPRINT.encode()
    async with semaphore:
        try:
            async with client.stream("GET", base_url + "/") as response:
                if response.status_code != 200:
                    return None
                # Read only the first bytes: some LAN devices stream endlessly at "/".
                head = b""
                async for chunk in response.aiter_bytes():
                    head += chunk
                    if len(head) >= len(fingerprint):
                        break
        except Exception:
            return None
        if not head.startswith(fingerprint):
            return None

        models: List[str] = []
        if with_models:
            try:
                tags = await client.get(base_url + "/api/tags", timeout=max(timeout, 3.0))
                models = [str(m.get("name")) for m in tags.json().get("models", [])]
            except Exception:
                pass
    return Server(host=host, port=port, base_url=base_url, models=models)


def _sort_key(server: Server) -> Tuple[int, Any]:
    try:
        return (0, ipaddress.ip_address(server.host))
    except ValueError:
        return (1, server.host)


async def adiscover(
    *,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    subnets: Optional[Sequence[str]] = None,
    hosts: Optional[Sequence[str]] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    with_models: bool = True,
    first: bool = False,
    http_client: Optional[httpx.AsyncClient] = None,
) -> List[Server]:
    """Scan the network for Thinai apps.

    Args:
        port: Port the app serves on (the app's Server page shows it).
        timeout: Per-host probe timeout in seconds.
        subnets: CIDRs to scan, e.g. ``["192.168.1.0/24"]``. Defaults to this machine's /24s.
        hosts: Exact hosts to probe instead of scanning subnets.
        concurrency: Maximum simultaneous probes.
        with_models: Also fetch each server's installed model names.
        first: Stop as soon as one server answers.
    """
    targets = list(hosts) if hosts is not None else candidate_hosts(subnets)
    if not targets:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        trust_env=False,  # never route LAN probes through an HTTP proxy
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=0),
    )
    tasks = [
        asyncio.ensure_future(_probe(client, h, port, semaphore, with_models, timeout))
        for h in targets
    ]
    found: List[Server] = []
    try:
        for next_done in asyncio.as_completed(tasks):
            server = await next_done
            if server is not None:
                found.append(server)
                if first:
                    break
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if http_client is None:
            await client.aclose()
    return sorted(found, key=_sort_key)


def discover(
    *,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    subnets: Optional[Sequence[str]] = None,
    hosts: Optional[Sequence[str]] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    with_models: bool = True,
    first: bool = False,
) -> List[Server]:
    """Blocking version of :func:`adiscover`. Safe to call from Jupyter."""
    return _run_sync(
        adiscover(
            port=port,
            timeout=timeout,
            subnets=subnets,
            hosts=hosts,
            concurrency=concurrency,
            with_models=with_models,
            first=first,
        )
    )


_NOT_FOUND_HELP = (
    "No Thinai server found on port {port}. In the Thinai app open Server, press Start, "
    "turn on 'Share on local network', and make sure this computer is on the same Wi-Fi. "
    "Or connect directly: Thinai(host='192.168.x.y'), or set THINAI_HOST."
)


async def afind_server(
    *, port: int = DEFAULT_PORT, timeout: float = DEFAULT_PROBE_TIMEOUT
) -> Server:
    """Return the first Thinai server found (this machine first, then the LAN)."""
    servers = await adiscover(
        port=port,
        timeout=timeout,
        hosts=["127.0.0.1", *candidate_hosts()],
        with_models=False,
        first=True,
    )
    if not servers:
        raise DiscoveryError(_NOT_FOUND_HELP.format(port=port))
    return servers[0]


def find_server(*, port: int = DEFAULT_PORT, timeout: float = DEFAULT_PROBE_TIMEOUT) -> Server:
    """Blocking version of :func:`afind_server`."""
    return _run_sync(afind_server(port=port, timeout=timeout))


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (Jupyter, async app): run on a separate thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
