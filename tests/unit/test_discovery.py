import httpx
import pytest

from thinai import DiscoveryError, adiscover, afind_server, discover
from thinai.discovery import candidate_hosts, local_ipv4_addresses


def test_candidate_hosts_from_subnets() -> None:
    assert candidate_hosts(["192.168.1.0/30"]) == ["192.168.1.1", "192.168.1.2"]
    assert candidate_hosts(["192.168.1.36/32", "192.168.1.36"]) == ["192.168.1.36"]
    assert len(candidate_hosts(["192.168.1.99/24"])) == 254


def test_candidate_hosts_rejects_huge_subnet() -> None:
    with pytest.raises(ValueError, match="too large"):
        candidate_hosts(["10.0.0.0/8"])


def test_candidate_hosts_defaults_to_local_24(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("thinai.discovery.local_ipv4_addresses", lambda: ["192.168.1.34"])
    hosts = candidate_hosts()
    assert hosts[0] == "192.168.1.1" and hosts[-1] == "192.168.1.254"


def test_local_addresses_are_private() -> None:
    import ipaddress

    for address in local_ipv4_addresses():
        assert ipaddress.IPv4Address(address).is_private


def _lan(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "10.0.0.1":
        return httpx.Response(200, text="Ollama is running")
    if host == "10.0.0.2":
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "lfm2.5-350m-q8_0"}]})
        return httpx.Response(200, text="Thinai is running. See /api/tags or /v1/models.")
    if host == "10.0.0.4":
        return httpx.Response(200, text="Thinai is running.")
    raise httpx.ConnectError("unreachable")


async def test_adiscover_matches_fingerprint() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_lan)) as client:
        servers = await adiscover(
            hosts=["10.0.0.4", "10.0.0.3", "10.0.0.2", "10.0.0.1"], http_client=client
        )
    assert [s.base_url for s in servers] == ["http://10.0.0.2:11434", "http://10.0.0.4:11434"]
    assert servers[0].models == ["lfm2.5-350m-q8_0"]


async def test_adiscover_first() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_lan)) as client:
        servers = await adiscover(hosts=["10.0.0.2", "10.0.0.4"], first=True, http_client=client)
    assert len(servers) == 1


async def test_discover_inside_running_loop() -> None:
    assert discover(hosts=[]) == []


async def test_afind_server_raises_with_help(monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing(**kwargs: object) -> list:
        return []

    monkeypatch.setattr("thinai.discovery.adiscover", nothing)
    with pytest.raises(DiscoveryError, match="Share on local network"):
        await afind_server()
