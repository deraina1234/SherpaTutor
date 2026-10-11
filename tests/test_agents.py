import asyncio

import pytest

from agents import IntegrityAgent, LearningAgent, MomentumAgent
from contracts import Argument, JevNoul, Position, argument_id
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


class GroundingJev(FakeJevClient):
    """FakeJevClient whose grounding checks pass unless the draft mentions `ungrounded`."""

    def __init__(self, ungrounded: str = "indentation", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ungrounded = ungrounded

    async def noul(self, state: dict, question: str):
        if "argument_text" in state:
            self.calls.append(("noul", state, question))
            text = state["argument_text"]
            if "will read" in question:  # the reveals-the-fix check
                return JevNoul(probability=0.9 if "missing check" in text else 0.1, confidence=0.9)
            return JevNoul(probability=0.1 if self.ungrounded in text else 0.9, confidence=0.9)
        return await super().noul(state, question)


def scripted_writer(reply: str | Exception):
    prompts: list[str] = []

    async def write(prompt: str) -> str | None:
        prompts.append(prompt)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return write, prompts


def test_propose_words_arguments_with_writer_and_checks_them_with_jev() -> None:
    reply = '```json\n{"repeated_error": "Case 4 failed the same way 3 runs in a row.", "long_idle": "No edits for 240 seconds."}\n```'
    writer, prompts = scripted_writer(reply)
    jev = GroundingJev(score_value=2, noul_probability=0.1)
    position = asyncio.run(MomentumAgent(jev, writer).propose(MOMENTUM_SLICE))
    texts = {a.claim: a.text for a in position.arguments}
    assert texts == {"repeated_error": "Case 4 failed the same way 3 runs in a row.", "long_idle": "No edits for 240 seconds."}
    assert len(prompts) == 1 and '"repeated_error"' in prompts[0] and "240" in prompts[0]
    checked = [state["argument_text"] for kind, state, _ in jev.calls if "argument_text" in state]
    assert sorted(checked) == sorted([*texts.values(), *texts.values()])  # grounded? and reveals the fix?


def test_propose_drops_drafts_that_fail_the_grounding_check() -> None:
    writer, _ = scripted_writer('{"repeated_error": "Your indentation is wrong on line 3.", "long_idle": "No edits for 240 seconds."}')
    position = asyncio.run(MomentumAgent(GroundingJev(score_value=2, noul_probability=0.1), writer).propose(MOMENTUM_SLICE))
    texts = {a.claim: a.text for a in position.arguments}
    assert texts == {"repeated_error": None, "long_idle": "No edits for 240 seconds."}


@pytest.mark.parametrize("reply", ["not json at all", '{"repeated_error": 3}', '["a list"]', RuntimeError("model down")])
def test_propose_falls_back_to_tags_when_the_writer_fails(reply) -> None:
    writer, _ = scripted_writer(reply)
    position = asyncio.run(MomentumAgent(GroundingJev(score_value=2, noul_probability=0.1), writer).propose(MOMENTUM_SLICE))
    assert position.reason_tags == ["repeated_error", "long_idle"]
    assert all(a.text is None for a in position.arguments)


def test_integrity_never_sends_its_slice_to_the_writer() -> None:
    writer, prompts = scripted_writer('{"solution_leak_risk": "anything"}')
    position = asyncio.run(IntegrityAgent(FakeJevClient(score_value=4, noul_probability=0.9), writer).propose(INTEGRITY_SLICE))
    assert prompts == []
    assert position.arguments[0].text is None


def test_revise_judges_the_peers_written_argument_and_words_its_rebuttal() -> None:
    writer, prompts = scripted_writer('{"momentum.r0.long_idle": "They ran the tests 40 seconds ago."}')
    jev = GroundingJev(score_value=1, noul_probability=0.1)
    agent = LearningAgent(jev, writer)
    momentum = argued("momentum", 3, "long_idle", 1)
    momentum = Position(
        agent="momentum", round=0, level=3, confidence=0.8, reason_tags=["long_idle"],
        arguments=[Argument(**{**momentum.arguments[0].to_dict(), "text": "No edits for 240 seconds."})],
    )

    async def run() -> Position:
        await agent.propose(LEARNING_SLICE)
        return await agent.revise(LEARNING_SLICE, [momentum], 1)

    revised = asyncio.run(run())
    judged = [state for kind, state, _ in jev.calls if "peer_argument" in state]
    assert judged[0]["peer_argument"]["claim"] == "No edits for 240 seconds."
    rebuttal = revised.arguments[-1]
    assert rebuttal.attacks == "momentum.r0.long_idle"
    assert rebuttal.text == "They ran the tests 40 seconds ago."
    assert "does not hold" in prompts[-1] and "No edits for 240 seconds." in prompts[-1]


def test_revise_drops_a_reply_that_just_repeats_the_peer() -> None:
    writer, _ = scripted_writer('{"momentum.r0.long_idle": "No edits for 240 seconds!"}')
    agent = LearningAgent(GroundingJev(score_value=1, noul_probability=0.1), writer)
    momentum = argued("momentum", 3, "long_idle", 1)
    momentum = Position(
        agent="momentum", round=0, level=3, confidence=0.8, reason_tags=["long_idle"],
        arguments=[Argument(**{**momentum.arguments[0].to_dict(), "text": "No edits for 240 seconds."})],
    )

    async def run() -> Position:
        await agent.propose(LEARNING_SLICE)
        return await agent.revise(LEARNING_SLICE, [momentum], 1)

    rebuttal = asyncio.run(run()).arguments[-1]
    assert rebuttal.attacks == "momentum.r0.long_idle" and rebuttal.text is None


def test_propose_drops_drafts_that_point_at_the_fix() -> None:
    writer, _ = scripted_writer('{"repeated_error": "There is a missing check after the loop.", "long_idle": "No edits for 240 seconds."}')
    position = asyncio.run(MomentumAgent(GroundingJev(score_value=2, noul_probability=0.1), writer).propose(MOMENTUM_SLICE))
    assert {a.claim: a.text for a in position.arguments} == {"repeated_error": None, "long_idle": "No edits for 240 seconds."}
