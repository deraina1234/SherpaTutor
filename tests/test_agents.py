import asyncio

import pytest

from agents import IntegrityAgent, LearningAgent, MomentumAgent
from contracts import Argument, Position, argument_id
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


def argued(agent: str, level: int, claim: str, direction: int) -> Position:
    argument = Argument(
        id=argument_id(agent, 0, claim), agent=agent, round=0, claim=claim, direction=direction, evidence={"n": 3}
    )
    return Position(agent=agent, round=0, level=level, confidence=0.8, reason_tags=[claim], arguments=[argument])


@pytest.mark.parametrize(
    ("score", "peer_level", "claim", "direction", "expected"),
    [(1, 3, "repeated_error", 1, 2), (2, 0, "active_progress", -1, 1)],
)
def test_revise_moves_one_step_toward_peer_when_its_argument_holds(
    score: int, peer_level: int, claim: str, direction: int, expected: int
) -> None:
    agent = LearningAgent(FakeJevClient(score_value=score, noul_probability=0.9))

    async def run() -> Position:
        await agent.propose(LEARNING_SLICE)
        return await agent.revise(LEARNING_SLICE, [argued("momentum", peer_level, claim, direction)], 1)

    revised = asyncio.run(run())
    assert (revised.level, revised.round, revised.confidence) == (expected, 1, 0.9)
    response = revised.arguments[-1]
    assert response.evidence == {"responds_to": f"momentum.r0.{claim}", "holds": 0.9}
    assert response.attacks is None and response.direction == direction


def test_revise_rebuts_a_weak_argument_and_keeps_its_level() -> None:
    agent = LearningAgent(FakeJevClient(score_value=1, noul_probability=0.1))

    async def run() -> Position:
        await agent.propose(LEARNING_SLICE)
        return await agent.revise(LEARNING_SLICE, [argued("momentum", 3, "long_idle", 1)], 1)

    revised = asyncio.run(run())
    assert (revised.level, revised.round) == (1, 1)
    rebuttal = revised.arguments[-1]
    assert (rebuttal.attacks, rebuttal.direction) == ("momentum.r0.long_idle", 0)


def test_revise_judges_peer_arguments_once_against_own_observations() -> None:
    jev = FakeJevClient(score_value=1, noul_probability=0.5)
    agent = LearningAgent(jev)
    momentum = argued("momentum", 3, "repeated_error", 1)

    async def run() -> tuple[Position, Position, Position]:
        mine = await agent.propose(LEARNING_SLICE)
        first = await agent.revise(LEARNING_SLICE, [mine, momentum], 1)
        return mine, first, await agent.revise(LEARNING_SLICE, [mine, momentum], 2)

    mine, first, second = asyncio.run(run())
    assert first is mine and second is mine  # undecided: neither moves nor rebuts
    judged = [state for kind, state, _ in jev.calls if kind == "noul" and "peer_argument" in state]
    assert len(judged) == 1
    assert judged[0]["peer_argument"]["agent"] == "momentum"
    assert judged[0]["peer_argument"]["wants"] == "more help"
    assert judged[0]["my_vote"]["level"] == 1
    assert judged[0]["observations"] == LEARNING_SLICE


def test_revise_ignores_arguments_that_do_not_pull_toward_the_peer() -> None:
    jev = FakeJevClient(score_value=1, noul_probability=0.9)
    agent = LearningAgent(jev)

    async def run() -> tuple[Position, Position]:
        first = await agent.propose(LEARNING_SLICE)
        return first, await agent.revise(LEARNING_SLICE, [argued("momentum", 3, "active_progress", -1)], 1)

    first, revised = asyncio.run(run())
    assert revised is first


def test_propose_attaches_evidence_backed_arguments() -> None:
    position = asyncio.run(MomentumAgent(FakeJevClient(score_value=2, noul_probability=0.9)).propose(MOMENTUM_SLICE))
    by_claim = {a.claim: a for a in position.arguments}
    assert by_claim["long_idle"].evidence["seconds_since_last_edit"] == 240
    assert by_claim["repeated_error"].direction == 1
    assert by_claim["learner_requested_help"].evidence["p_asked"] == 0.9


def test_voting_agents_never_score_or_raise_into_near_solution() -> None:
    jev = FakeJevClient(score_value=4, noul_probability=0.1, choice_selected="raise")
    agent = MomentumAgent(jev)

    async def run() -> tuple[Position, Position]:
        first = await agent.propose(MOMENTUM_SLICE)
        return first, await agent.revise(MOMENTUM_SLICE, [], 1)

    first, revised = asyncio.run(run())
    assert first.level == 3
    assert revised is first
    _, score_state, _ = jev.calls[0]
    assert len(score_state["help_levels"]) == 4


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
