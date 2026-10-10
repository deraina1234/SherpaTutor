"""Tests for deliberation loop, InProcBus, and stop rules."""

from __future__ import annotations

import asyncio
from typing import Any

from agents import IntegrityAgent, LearningAgent, MomentumAgent
from contracts import (
    Agent,
    HelpLevel,
    OutcomeKind,
    Position,
    SliceDict,
)
from deliberation import InProcBus, evaluate_stop_rules, run_deliberation
from jev import FakeJevClient
from logger import log_deliberation, read_deliberations


class StubAgent:
    """Deterministic agent that emits predefined proposals and revisions."""

    def __init__(self, name: str, positions: list[Position]) -> None:
        self.name = name
        self.positions = positions
        self._step = 0

    async def propose(self, slice: SliceDict) -> Position:
        pos = self.positions[self._step]
        self._step += 1
        return pos

    async def revise(self, slice: SliceDict, peers: list[Position], round: int) -> Position:
        if self._step < len(self.positions):
            pos = self.positions[self._step]
            self._step += 1
            return pos
        return self.positions[-1]


def make_pos(agent: str, round: int, level: int, cap: int | None = None) -> Position:
    return Position(
        agent=agent,
        round=round,
        level=level,
        confidence=0.85,
        reason_tags=[],
        cap=cap,
    )


def test_inproc_bus_pub_sub() -> None:
    async def run() -> None:
        bus = InProcBus()
        received_topic: list[dict[str, Any]] = []
        received_all: list[dict[str, Any]] = []

        async def sub_topic() -> None:
            async for msg in bus.subscribe("deliberation/round/0"):
                received_topic.append(msg)
                break

        async def sub_wildcard() -> None:
            async for msg in bus.subscribe("*"):
                received_all.append(msg)
                if len(received_all) == 2:
                    break

        task1 = asyncio.create_task(sub_topic())
        task2 = asyncio.create_task(sub_wildcard())
        await asyncio.sleep(0.01)

        await bus.publish("deliberation/round/0", {"round": 0, "status": "ok"})
        await bus.publish("deliberation/round/1", {"round": 1, "status": "ok"})

        await asyncio.wait_for(asyncio.gather(task1, task2), timeout=1.0)

        assert len(received_topic) == 1
        assert received_topic[0]["round"] == 0
        assert len(received_all) == 2

    asyncio.run(run())


def test_evaluate_stop_rules_consensus() -> None:
    rounds = [
        [
            make_pos("learning", 0, 2),
            make_pos("momentum", 0, 2),
            make_pos("integrity", 0, 2, cap=3),
        ]
    ]
    outcome, final_level, cap_applied, ask_learner = evaluate_stop_rules(rounds)
    assert outcome == OutcomeKind.CONSENSUS
    assert final_level == 2
    assert not cap_applied
    assert not ask_learner


def test_evaluate_stop_rules_soft_disagreement() -> None:
    rounds = [
        [
            make_pos("learning", 1, 1),
            make_pos("momentum", 1, 2),
            make_pos("integrity", 1, 2, cap=3),
        ]
    ]
    outcome, final_level, cap_applied, ask_learner = evaluate_stop_rules(rounds)
    assert outcome == OutcomeKind.SOFT_DISAGREEMENT
    assert final_level == 1  # picks the lower, least-intrusive level
    assert not cap_applied
    assert not ask_learner


def test_evaluate_stop_rules_deadlock() -> None:
    rounds = [
        [
            make_pos("learning", 2, 0),
            make_pos("momentum", 2, 3),
            make_pos("integrity", 2, 2, cap=3),
        ]
    ]
    outcome, final_level, cap_applied, ask_learner = evaluate_stop_rules(rounds)
    assert outcome == OutcomeKind.DEADLOCK
    assert final_level is None
    assert not cap_applied
    assert ask_learner


def test_evaluate_stop_rules_integrity_cap_clamping() -> None:
    rounds = [
        [
            make_pos("learning", 1, 2),
            make_pos("momentum", 1, 3),  # above cap=2
            make_pos("integrity", 1, 2, cap=2),
        ]
    ]
    # momentum's 3 clamped to 2 -> all become 2 -> consensus!
    outcome, final_level, cap_applied, ask_learner = evaluate_stop_rules(rounds)
    assert outcome == OutcomeKind.CONSENSUS
    assert final_level == 2
    assert cap_applied
    assert not ask_learner


def test_deliberation_round0_consensus_stops_early() -> None:
    async def run() -> None:
        agents: list[Agent] = [
            StubAgent("learning", [make_pos("learning", 0, 2)]),
            StubAgent("momentum", [make_pos("momentum", 0, 2)]),
            StubAgent("integrity", [make_pos("integrity", 0, 2, cap=3)]),
        ]
        slices = {"learning": {}, "momentum": {}, "integrity": {}}

        decision = await run_deliberation(agents, slices)
        assert decision.outcome == OutcomeKind.CONSENSUS
        assert decision.final_level == 2
        assert decision.rounds_used == 1  # Only round 0 needed
        assert len(decision.rounds) == 1

    asyncio.run(run())


def test_deliberation_converges_in_round_1() -> None:
    async def run() -> None:
        # Round 0: [1, 3, 2] -> conflict
        # Round 1: [2, 2, 2] -> consensus
        agents: list[Agent] = [
            StubAgent("learning", [make_pos("learning", 0, 1), make_pos("learning", 1, 2)]),
            StubAgent("momentum", [make_pos("momentum", 0, 3), make_pos("momentum", 1, 2)]),
            StubAgent("integrity", [make_pos("integrity", 0, 2, cap=2), make_pos("integrity", 1, 2, cap=2)]),
        ]
        slices = {"learning": {}, "momentum": {}, "integrity": {}}

        decision = await run_deliberation(agents, slices)
        assert decision.outcome == OutcomeKind.CONSENSUS
        assert decision.final_level == 2
        assert decision.rounds_used == 2  # round 0 and round 1
        assert len(decision.rounds) == 2

    asyncio.run(run())


def test_deliberation_unanimous_hold_stops_early() -> None:
    async def run() -> None:
        # Round 0: [1, 2, 2] (spread 1)
        # Round 1: all agents hold, returning same Position instances
        pos_l0 = make_pos("learning", 0, 1)
        pos_m0 = make_pos("momentum", 0, 2)
        pos_i0 = make_pos("integrity", 0, 2, cap=2)

        agents: list[Agent] = [
            StubAgent("learning", [pos_l0, pos_l0]),
            StubAgent("momentum", [pos_m0, pos_m0]),
            StubAgent("integrity", [pos_i0, pos_i0]),
        ]
        slices = {"learning": {}, "momentum": {}, "integrity": {}}

        decision = await run_deliberation(agents, slices, max_rounds=3)
        assert decision.outcome == OutcomeKind.SOFT_DISAGREEMENT
        assert decision.final_level == 1
        # Stopped after round 1 because everyone held, instead of spinning through rounds 2 and 3
        assert decision.rounds_used == 2
        assert len(decision.rounds) == 2

    asyncio.run(run())


def test_deliberation_with_real_agents_and_bus() -> None:
    async def run() -> None:
        jev = FakeJevClient(score_value=2.0, choice_selected="hold", confidence=0.8)
        agents: list[Agent] = [
            LearningAgent(jev),
            MomentumAgent(jev),
            IntegrityAgent(jev),
        ]
        slices: dict[str, SliceDict] = {
            "learning": {
                "learner_level": "beginner",
                "hint_history": [],
                "seconds_on_task": 300,
                "recent_progress": False,
            },
            "momentum": {
                "seconds_since_last_edit": 150,
                "repeated_error_count": 2,
                "failed_runs_since_progress": 2,
                "same_function_rewrites": 1,
                "learner_message": None,
            },
            "integrity": {
                "task_id": "two_sum",
                "reference_solution_summary": "hash map lookup",
                "policy_max_level": HelpLevel.POINT_TO_AREA,
            },
        }

        bus = InProcBus()
        published_events: list[dict[str, Any]] = []

        async def listener() -> None:
            async for msg in bus.subscribe("*"):
                published_events.append(msg)
                if "outcome" in msg:
                    break

        listen_task = asyncio.create_task(listener())
        await asyncio.sleep(0.01)

        decision = await run_deliberation(agents, slices, bus=bus)
        await asyncio.wait_for(listen_task, timeout=1.0)

        assert decision.outcome == OutcomeKind.CONSENSUS
        assert decision.final_level == 2
        assert len(published_events) >= 2  # round 0 + decision

    asyncio.run(run())


def test_logger_roundtrip(tmp_path) -> None:
    log_file = tmp_path / "sessions.jsonl"
    decision = evaluate_stop_rules([[make_pos("learning", 0, 1), make_pos("momentum", 0, 2)]])
    from contracts import Decision
    d = Decision(
        outcome=decision[0],
        final_level=decision[1],
        rounds=[[make_pos("learning", 0, 1), make_pos("momentum", 0, 2)]],
        cap_applied=decision[2],
        ask_learner=decision[3],
        rounds_used=1,
    )
    slices = {"learning": {"learner_level": "beginner"}, "momentum": {}}

    record = log_deliberation(
        filepath=log_file,
        session_id="sess_123",
        task_id="two_sum",
        slices=slices,
        decision=d,
        metadata={"user": "test_user"},
    )
    assert record["session_id"] == "sess_123"

    loaded = read_deliberations(log_file)
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "two_sum"
    assert loaded[0]["decision"]["outcome"] == "soft_disagreement"
    assert loaded[0]["decision"]["final_level"] == 1
