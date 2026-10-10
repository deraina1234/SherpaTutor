"""JevClient implementations.

TypeSafeJevClient calls the real TypeSafe API through the typesafe-sdk package and
adapts its answers to the shapes in contracts.py. FakeJevClient returns canned
answers so agents and the loop can be built and tested offline.
"""

from __future__ import annotations
from types import TracebackType
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, Score

from contracts import JevChoice, JevNoul, JevScore

_QUESTION_ID = "q"  # one question per request; ids are never sent to the model


def noul_confidence(probability: float) -> float:
    """Distance from a coin flip, 0..1. The API reports no confidence for nouls."""
    return abs(2 * probability - 1)


class TypeSafeJevClient:
    """JevClient backed by the TypeSafe API.

    Reads TYPESAFE_API_KEY from the environment unless api_key is given. Retries on
    429 and 529 are handled by the SDK. Use as an async context manager, or call
    aclose() when done.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: AsyncTypeSafeClient | None = None,  # inject a preconfigured SDK client, e.g. in tests
    ) -> None:
        self._client = client or AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout)

    async def __aenter__(self) -> TypeSafeJevClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def score(self, state: dict, question: str, levels: list[str]) -> JevScore:
        response = await self._ask(state, Score(instructions=question, criteria=levels))
        answer = response.scores[_QUESTION_ID]
        return JevScore(value=answer.score, probabilities=dict(answer.probabilities), confidence=answer.confidence)

    async def noul(self, state: dict, question: str) -> JevNoul:
        response = await self._ask(state, Noul(instructions=question))
        probability = response.nouls[_QUESTION_ID].noul
        return JevNoul(probability=probability, confidence=noul_confidence(probability))

    async def choice(self, state: dict, question: str, options: list[str]) -> JevChoice:
        response = await self._ask(state, Choice(instructions=question, criteria={option: None for option in options}))
        answer = response.choices[_QUESTION_ID]
        return JevChoice(selected=answer.choice, probabilities=dict(answer.probabilities), confidence=answer.confidence)

    async def _ask(self, state: dict, question: Noul | Choice | Score) -> Any:
        return await self._client.system_one(state=state, questions={_QUESTION_ID: question})


class FakeJevClient:
    """Canned JevClient for tests and offline dev. Records every call in `calls`."""

    def __init__(
        self,
        score_value: float = 2.0,
        noul_probability: float = 0.5,
        choice_selected: str = "hold",
        confidence: float = 0.9,
    ) -> None:
        self.score_value = score_value
        self.noul_probability = noul_probability
        self.choice_selected = choice_selected
        self.confidence = confidence
        self.calls: list[tuple[str, dict, str]] = []

    async def score(self, state: dict, question: str, levels: list[str]) -> JevScore:
        self.calls.append(("score", state, question))
        value = min(max(self.score_value, 0), len(levels) - 1)
        return JevScore(value=value, probabilities={round(value): 1.0}, confidence=self.confidence)

    async def noul(self, state: dict, question: str) -> JevNoul:
        self.calls.append(("noul", state, question))
        return JevNoul(probability=self.noul_probability, confidence=self.confidence)

    async def choice(self, state: dict, question: str, options: list[str]) -> JevChoice:
        self.calls.append(("choice", state, question))
        selected = self.choice_selected if self.choice_selected in options else options[0]
        probabilities = {option: 1.0 if option == selected else 0.0 for option in options}
        return JevChoice(selected=selected, probabilities=probabilities, confidence=self.confidence)


def _load_env_file(path: str) -> None:
    """Minimal KEY=VALUE loader for .env.local; existing environment variables win."""
    import os

    try:
        with open(path) as file:
            lines = file.read().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


if __name__ == "__main__":
    # Smoke test against the real API: python jev.py
    import asyncio

    _load_env_file(".env.local")

    async def main() -> None:
        state = {"learner_message": "I keep getting IndexError and I don't get why. Can you help?"}
        async with TypeSafeJevClient() as jev:
            noul, choice, score = await asyncio.gather(
                jev.noul(state, "Is `learner_message` the learner asking for help?"),
                jev.choice(state, "What kind of problem is the learner describing?", ["syntax", "runtime_error", "logic", "other"]),
                jev.score(
                    state,
                    "How frustrated does the learner sound in `learner_message`?",
                    ["calm and curious", "mildly frustrated", "very frustrated or ready to give up"],
                ),
            )
        print(noul)
        print(choice)
        print(score)

    asyncio.run(main())
