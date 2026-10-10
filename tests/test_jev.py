import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient

from contracts import JevChoice, JevNoul, JevScore
from jev import TypeSafeJevClient, noul_confidence

ANSWERS = {
    "noul": {"type": "noul", "noul": 0.9},
    "choice": {"type": "choice", "choice": "hold", "probabilities": {"lower": 0.1, "hold": 0.8, "raise": 0.1}, "confidence": 0.7},
    "score": {
        "type": "score",
        "score": 1.2,
        "legend": {"0": "none", "1": "nudge", "2": "hint"},
        "probabilities": {"0": 0.0, "1": 0.8, "2": 0.2},
        "confidence": 0.6,
    },
}


def fake_api(requests: list[dict]) -> TypeSafeJevClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        question = body["questions"]["q"]
        return httpx2.Response(
            200,
            json={"model": "jev-test", "answers": {"q": ANSWERS[question["type"]]}, "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    sdk = AsyncTypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler))
    return TypeSafeJevClient(client=sdk)


def test_noul_adapts_and_derives_confidence() -> None:
    requests: list[dict] = []
    result = asyncio.run(fake_api(requests).noul({"x": 1}, "Is it?"))
    assert result == JevNoul(probability=0.9, confidence=pytest.approx(0.8))
    assert requests[0]["state"] == {"x": 1}
    assert requests[0]["questions"]["q"]["instructions"] == "Is it?"


def test_choice_sends_options_as_criteria() -> None:
    requests: list[dict] = []
    result = asyncio.run(fake_api(requests).choice({}, "Move?", ["lower", "hold", "raise"]))
    assert result == JevChoice(selected="hold", probabilities={"lower": 0.1, "hold": 0.8, "raise": 0.1}, confidence=0.7)
    assert requests[0]["questions"]["q"]["criteria"] == {"lower": None, "hold": None, "raise": None}


def test_score_sends_levels_and_int_keys_probabilities() -> None:
    requests: list[dict] = []
    result = asyncio.run(fake_api(requests).score({}, "How much?", ["none", "nudge", "hint"]))
    assert result == JevScore(value=1.2, probabilities={0: 0.0, 1: 0.8, 2: 0.2}, confidence=0.6)
    assert requests[0]["questions"]["q"]["criteria"] == ["none", "nudge", "hint"]


@pytest.mark.parametrize(("probability", "expected"), [(0.5, 0.0), (1.0, 1.0), (0.0, 1.0), (0.75, 0.5)])
def test_noul_confidence(probability: float, expected: float) -> None:
    assert noul_confidence(probability) == pytest.approx(expected)
