import httpx
import pytest

from thinai import Server, __main__


def test_cli_health(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    real_client = httpx.Client

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/healthz"
            return httpx.Response(200, json={"status": "ok"})

        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("thinai.client.httpx.Client", fake_client)
    assert __main__.main(["health", "--host", "abc.trycloudflare.com"]) == 0
    assert "https://abc.trycloudflare.com: ok" in capsys.readouterr().out


def test_cli_discover_marks_protected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = {}

    def fake_discover(**kwargs: object) -> list:
        seen.update(kwargs)
        return [Server("10.0.0.5", 11434, "http://10.0.0.5:11434", [], auth_required=True)]

    monkeypatch.setattr(__main__, "discover", fake_discover)
    monkeypatch.setenv("THINAI_API_KEY", "envtok")
    assert __main__.main(["discover"]) == 0
    assert seen["api_key"] == "envtok"
    assert "(token required)" in capsys.readouterr().out
