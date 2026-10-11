import asyncio
import json

import httpx
import pytest

import hints
from hints import _nemotron_complete, write_hint_async

CODE = "def second_largest(nums):\n    nums.sort()\n    return nums[-2]\n"


@pytest.fixture
def nemotron_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://nemotron.test/v1/")
    monkeypatch.setenv("NEMOTRON_MODEL", "nvidia/test-model")
    monkeypatch.setenv("NEMOTRON_API_KEY", "nvapi-test")
    monkeypatch.setenv("NEMOTRON_SYSTEM", "/no_think")


def completion(content: str | None, status: int = 200):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    return httpx.MockTransport(handler), requests


def test_disabled_without_env_returns_none() -> None:
    assert asyncio.run(_nemotron_complete("hi")) is None


def test_sends_openai_chat_request_and_strips_reasoning(nemotron_env) -> None:
    transport, requests = completion("<think>they sorted, dupes...</think>\n  Watch what happens to duplicates.  ")
    text = asyncio.run(_nemotron_complete("write a hint", transport=transport))
    assert text == "Watch what happens to duplicates."

    request = requests[0]
    assert str(request.url) == "http://nemotron.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer nvapi-test"
    body = json.loads(request.content)
    assert body["model"] == "nvidia/test-model"
    assert body["messages"] == [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "write a hint"},
    ]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(("content", "status"), [("", 200), ("<think>only thinking</think>", 200), ("x", 500)])
def test_failures_return_none(nemotron_env, content: str, status: int) -> None:
    transport, _ = completion(content, status)
    assert asyncio.run(_nemotron_complete("p", transport=transport)) is None


def test_hint_prefers_nemotron_then_falls_back_to_template(nemotron_env, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_nemotron(prompt: str, transport=None) -> str | None:
        return "Nemotron hint"

    monkeypatch.setattr(hints, "_nemotron_complete", fake_nemotron)
    assert asyncio.run(write_hint_async(CODE, "Case 2: expected 4, got 5", 2)).text == "Nemotron hint"

    async def failing_nemotron(prompt: str, transport=None) -> str | None:
        return None

    monkeypatch.setattr(hints, "_nemotron_complete", failing_nemotron)
    # No Gemini key in tests, so the template is next.
    template = asyncio.run(write_hint_async(CODE, "Case 2: expected 4, got 5", 2)).text
    assert template and template != "Nemotron hint"
