import asyncio

import pytest

from agents import IntegrityAgent, LearningAgent, MomentumAgent
from contracts import Position
from jev import FakeJevClient

LEARNING_SLICE = {
    "learner_level": "beginner",
    "hint_history": [{"level": 1, "seconds_ago": 60}],
    "seconds_on_task": 600,
    "recent_progress": False,
}
MOMENTUM_SLICE = {
    "seconds_since_last_edit": 240,
    "repeated_error_count": 3,
    "failed_runs_since_progress": 4,
    "same_function_rewrites": 2,
    "learner_message": "i'm stuck, help?",
}
INTEGRITY_SLICE = {
    "task_id": "fizzbuzz",
    "reference_solution_summary": "loop 1..n and check % 15, % 3, % 5 in that order",
    "policy_max_level": 3,
}


def peer(agent: str, level: int, tags: list[str]) -> Position:
    return Position(agent=agent, round=0, level=level, confidence=0.8, reason_tags=tags)


def test_learning_propose_tags_come_from_code_and_jev() -> None:
    position = asyncio.run(LearningAgent(FakeJevClient(score_value=1.2, noul_probability=0.9)).propose(LEARNING_SLICE))
    assert (position.agent, position.round, position.level) == ("learning", 0, 1)
    assert position.reason_tags == ["concept_gap", "recent_hint_given"]
    assert position.cap is None


def test_momentum_propose_tags() -> None:
    position = asyncio.run(MomentumAgent(FakeJevClient(score_value=3, noul_probability=0.9)).propose(MOMENTUM_SLICE))
    assert position.level == 3
    assert position.reason_tags == ["repeated_error", "long_idle", "learner_requested_help"]


def test_momentum_skips_request_question_without_message() -> None:
    jev = FakeJevClient()
    asyncio.run(MomentumAgent(jev).propose({**MOMENTUM_SLICE, "learner_message": None}))
    assert [kind for kind, _, _ in jev.calls] == ["score"]


def test_integrity_caps_at_policy_and_tightens_on_leak_risk() -> None:
    no_leak = asyncio.run(IntegrityAgent(FakeJevClient(score_value=4, noul_probability=0.1)).propose(INTEGRITY_SLICE))
    assert (no_leak.level, no_leak.cap) == (3, 3)
    leak = asyncio.run(IntegrityAgent(FakeJevClient(score_value=4, noul_probability=0.9)).propose(INTEGRITY_SLICE))
    assert (leak.level, leak.cap, leak.reason_tags) == (2, 2, ["solution_leak_risk"])


def test_revise_hold_returns_same_position() -> None:
    agent = LearningAgent(FakeJevClient(score_value=1, choice_selected="hold"))

    async def run() -> tuple[Position, Position]:
        first = await agent.propose(LEARNING_SLICE)
        return first, await agent.revise(LEARNING_SLICE, [peer("momentum", 3, ["repeated_error"])], 1)

    first, revised = asyncio.run(run())
    assert revised is first


@pytest.mark.parametrize(("selected", "expected"), [("raise", 2), ("lower", 0)])
def test_revise_moves_one_step(selected: str, expected: int) -> None:
    jev = FakeJevClient(score_value=1, choice_selected=selected)
    agent = LearningAgent(jev)

    async def run() -> Position:
        await agent.propose(LEARNING_SLICE)
        return await agent.revise(LEARNING_SLICE, [peer("momentum", 3, ["repeated_error"])], 1)

    revised = asyncio.run(run())
    assert (revised.level, revised.round) == (expected, 1)


def test_revise_sends_peers_but_not_self_to_jev() -> None:
    jev = FakeJevClient(score_value=1, choice_selected="raise")
    agent = LearningAgent(jev)

    async def run() -> None:
        mine = await agent.propose(LEARNING_SLICE)
        await agent.revise(LEARNING_SLICE, [mine, peer("momentum", 3, ["repeated_error"])], 1)

    asyncio.run(run())
    kind, state, _ = jev.calls[-1]
    assert kind == "choice"
    assert [p["agent"] for p in state["peer_positions"]] == ["momentum"]
    assert state["my_position"]["level"] == 1


def test_revise_holds_on_low_confidence() -> None:
    agent = LearningAgent(FakeJevClient(score_value=1, choice_selected="raise", confidence=0.3))

    async def run() -> tuple[Position, Position]:
        first = await agent.propose(LEARNING_SLICE)
        return first, await agent.revise(LEARNING_SLICE, [], 1)

    first, revised = asyncio.run(run())
    assert revised is first


def test_revise_holds_at_bounds_and_integrity_never_exceeds_cap() -> None:
    low = LearningAgent(FakeJevClient(score_value=0, choice_selected="lower"))
    capped = IntegrityAgent(FakeJevClient(score_value=4, noul_probability=0.9, choice_selected="raise"))

    async def run(agent, slice) -> tuple[Position, Position]:
        first = await agent.propose(slice)
        return first, await agent.revise(slice, [], 1)

    first, revised = asyncio.run(run(low, LEARNING_SLICE))
    assert revised is first
    first, revised = asyncio.run(run(capped, INTEGRITY_SLICE))
    assert revised is first and revised.level == revised.cap == 2


def test_revise_before_propose_raises() -> None:
    with pytest.raises(RuntimeError):
        asyncio.run(LearningAgent(FakeJevClient()).revise(LEARNING_SLICE, [], 1))


def test_integrity_revise_holds_without_calling_jev() -> None:
    jev = FakeJevClient(score_value=3, noul_probability=0.1, choice_selected="lower")
    agent = IntegrityAgent(jev)

    async def run() -> tuple[Position, Position]:
        first = await agent.propose(INTEGRITY_SLICE)
        return first, await agent.revise(INTEGRITY_SLICE, [peer("momentum", 3, ["repeated_error"])], 1)

    first, revised = asyncio.run(run())
    assert revised is first
    assert [kind for kind, _, _ in jev.calls] == ["score", "noul"]  # propose only


def test_integrity_revise_before_propose_raises() -> None:
    with pytest.raises(RuntimeError):
        asyncio.run(IntegrityAgent(FakeJevClient()).revise(INTEGRITY_SLICE, [], 1))
