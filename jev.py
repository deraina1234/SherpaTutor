"""JevClient implementations.

FakeJevClient returns canned answers so agents and the loop can be built and tested
offline. The real TypeSafe-backed client goes here too and adapts to the shapes in
contracts.py.
"""

from __future__ import annotations

from contracts import JevChoice, JevNoul, JevScore


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

    async def score(self, state: dict, question: str, scale: tuple[int, int]) -> JevScore:
        self.calls.append(("score", state, question))
        low, high = scale
        value = min(max(self.score_value, low), high)
        return JevScore(value=value, probabilities={round(value): 1.0}, confidence=self.confidence)

    async def noul(self, state: dict, question: str) -> JevNoul:
        self.calls.append(("noul", state, question))
        return JevNoul(probability=self.noul_probability, confidence=self.confidence)

    async def choice(self, state: dict, question: str, options: list[str]) -> JevChoice:
        self.calls.append(("choice", state, question))
        selected = self.choice_selected if self.choice_selected in options else options[0]
        probabilities = {option: 1.0 if option == selected else 0.0 for option in options}
        return JevChoice(selected=selected, probabilities=probabilities, confidence=self.confidence)
