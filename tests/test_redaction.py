import asyncio

import pytest

from contracts import Argument, Decision, OutcomeKind, Position, argument_id
from deliberation import InProcBus, run_deliberation
from logger import log_deliberation, read_deliberations
from redaction import REDACTED, redact, secret_values

SECRET = "loop 1..n, check % 15 before % 3 and % 5"
SLICES = {
    "learning": {"learner_level": "beginner"},
    "integrity": {"task_id": "fizzbuzz", "reference_solution_summary": SECRET, "policy_max_level": 3},
}


def leaky_position(agent: str = "integrity") -> Position:
    """A Position that smuggles the secret value under a non-secret key."""
    return Position(
        agent=agent,
        round=0,
        level=2,
        confidence=0.8,
        reason_tags=["solution_leak_risk"],
        arguments=[
            Argument(
                id=argument_id(agent, 0, "solution_leak_risk"),
                agent=agent,
                round=0,
                claim="solution_leak_risk",
                direction=-1,
                evidence={"note": f"answer is: {SECRET}"},
            )
        ],
    )


class StubAgent:
    def __init__(self, position: Position) -> None:
        self.name = position.agent
        self._position = position

    async def propose(self, slice: dict) -> Position:
        return self._position

    async def revise(self, slice: dict, peers: list[Position], round: int) -> Position:
        return self._position


def test_argument_rejects_secret_evidence_keys() -> None:
    with pytest.raises(ValueError):
        Argument(
            id="integrity.r0.solution_leak_risk",
            agent="integrity",
            round=0,
            claim="solution_leak_risk",
            direction=-1,
            evidence={"reference_solution_summary": SECRET},
        )


def test_secret_values_skips_missing_and_empty() -> None:
    assert secret_values(SLICES) == [SECRET]
    assert secret_values({"integrity": {"reference_solution_summary": "  "}, "learning": {}}) == []


def test_redact_is_recursive_and_does_not_mutate() -> None:
    original = {"a": [f"x {SECRET} y", ("t", SECRET)], SECRET: {"n": 1}}
    result = redact(original, [SECRET])
    assert result == {"a": [f"x {REDACTED} y", ("t", REDACTED)], REDACTED: {"n": 1}}
    assert original["a"][0] == f"x {SECRET} y"


def test_redact_without_secrets_is_identity() -> None:
    value = {"a": "b"}
    assert redact(value, []) is value


def test_logger_redacts_secret_value_anywhere(tmp_path) -> None:
    log_file = tmp_path / "sessions.jsonl"
    decision = Decision(
        outcome=OutcomeKind.CONSENSUS,
        final_level=2,
        rounds=[[leaky_position()]],
        cap_applied=False,
        ask_learner=False,
        rounds_used=1,
    )
    log_deliberation(log_file, "sess", "fizzbuzz", SLICES, decision, metadata={"debug": SECRET})
    assert SECRET not in log_file.read_text()
    record = read_deliberations(log_file)[0]
    assert "reference_solution_summary" not in record["slices"]["integrity"]
    assert record["metadata"]["debug"] == REDACTED


def test_deliberation_bus_messages_are_redacted() -> None:
    async def run() -> list[dict]:
        bus = InProcBus()
        sub = bus.subscribe("*")
        agents = [StubAgent(leaky_position()), StubAgent(Position("learning", 0, 2, 0.8, []))]
        decision = await run_deliberation(agents, SLICES, bus=bus)
        # The returned Decision is internal and unredacted; only published messages are scrubbed.
        assert decision.rounds[0][0].arguments[0].evidence["note"].endswith(SECRET)
        messages = []
        while not sub.queue.empty():
            messages.append(await anext(sub))
        return messages

    messages = asyncio.run(run())
    assert messages
    assert all(SECRET not in str(message) for message in messages)
