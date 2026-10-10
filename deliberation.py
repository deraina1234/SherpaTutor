"""Sherpa deliberation loop, InProcBus, and stop rules.

Implements the multi-agent deliberation protocol described in Section 8 of the
Spotter design doc:
- Round 0: Each peer agent proposes an initial position.
- Rounds 1..MAX_ROUNDS: Agents revise against peer positions.
- Stop rules:
    * Consensus (spread 0): Act at that level.
    * Soft disagreement (spread 1): Act at the lower level (least intrusive action).
    * Deadlock (spread >= 2): Withhold automatic hint; ask learner ("Want a nudge?").
    * Integrity cap: Clamp any level above Integrity's cap; set cap_applied = True.
    * Unanimous hold: Stop early if no agent changes position in a round.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from contracts import (
    MAX_ROUNDS,
    Agent,
    Bus,
    Decision,
    OutcomeKind,
    Position,
    SliceDict,
)
from redaction import RedactingBus, secret_values


_UNSUBSCRIBED = object()


class Subscription:
    """An active subscription registered immediately upon creation."""

    def __init__(self, bus: InProcBus, topic: str, queue: asyncio.Queue[Any]) -> None:
        self.bus = bus
        self.topic = topic
        self.queue = queue
        self._closed = False

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed and self.queue.empty():
            raise StopAsyncIteration
        msg = await self.queue.get()
        if msg is _UNSUBSCRIBED:
            raise StopAsyncIteration
        return msg

    def unsubscribe(self) -> None:
        if not self._closed:
            self._closed = True
            self.bus._remove_subscriber(self.topic, self.queue)
            self.queue.put_nowait(_UNSUBSCRIBED)

    async def aclose(self) -> None:
        self.unsubscribe()


class InProcBus:
    """In-process async pub/sub bus implementing contracts.Bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._wildcard_subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def _remove_subscriber(self, topic: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if topic == "*":
            if queue in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(queue)
        else:
            if topic in self._subscribers and queue in self._subscribers[topic]:
                self._subscribers[topic].remove(queue)

    async def publish(self, topic: str, msg: dict[str, Any]) -> None:
        """Publish a message to a topic."""
        for queue in list(self._subscribers.get(topic, [])):
            await queue.put(msg)
        for queue in list(self._wildcard_subscribers):
            await queue.put(msg)

    def subscribe(self, topic: str = "*") -> AsyncIterator[dict[str, Any]]:
        """Subscribe immediately to messages on a topic ('*' receives all topics)."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if topic == "*":
            self._wildcard_subscribers.append(queue)
        else:
            self._subscribers.setdefault(topic, []).append(queue)

        return Subscription(self, topic, queue)


def evaluate_stop_rules(
    rounds: list[list[Position]],
) -> tuple[OutcomeKind, int | None, bool, bool]:
    """Apply the protocol stop rules to the deliberation rounds.

    Integrity sets a maximum safe ceiling (cap), not a help request. Therefore,
    positions with a cap set are excluded from the spread calculation and are
    only used to clamp voting agents' proposals.

    Returns:
        (outcome, final_level, cap_applied, ask_learner)
    """
    if not rounds:
        raise ValueError("Cannot evaluate stop rules with empty rounds")

    last_positions = rounds[-1]

    # Find Integrity cap if specified by any agent (typically Integrity)
    cap: int | None = None
    for p in last_positions:
        if p.cap is not None:
            cap = p.cap if cap is None else min(cap, p.cap)

    # Active voting agents are those proposing help needs (where cap is None)
    voting_positions = [p for p in last_positions if p.cap is None]
    if not voting_positions:
        # Fallback if only capped agents are present
        voting_positions = last_positions

    # Check if clamping is required on any voting agent's proposal
    cap_applied = False
    effective_levels: list[int] = []
    for p in voting_positions:
        if cap is not None and p.level > cap:
            cap_applied = True
            effective_levels.append(cap)
        else:
            effective_levels.append(p.level)

    spread = max(effective_levels) - min(effective_levels)

    if spread == 0:
        outcome = OutcomeKind.CONSENSUS
        final_level = effective_levels[0]
        ask_learner = False
    elif spread == 1:
        outcome = OutcomeKind.SOFT_DISAGREEMENT
        final_level = min(effective_levels)
        ask_learner = False
    else:
        outcome = OutcomeKind.DEADLOCK
        final_level = None
        ask_learner = True

    return outcome, final_level, cap_applied, ask_learner


async def run_deliberation(
    agents: list[Agent],
    slices: dict[str, SliceDict],
    bus: Bus | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> Decision:
    """Run an end-to-end deliberation between peer agents.

    Args:
        agents: Fresh instances of peer agents (e.g. Learning, Momentum, Integrity).
        slices: Map of agent name to private slice dict.
        bus: Optional Bus implementation for publishing round messages.
        max_rounds: Maximum revision rounds (defaults to contracts.MAX_ROUNDS).
    """
    if not agents:
        raise ValueError("At least one agent is required for deliberation")

    for agent in agents:
        if agent.name not in slices:
            raise KeyError(f"Missing slice for agent '{agent.name}'")

    # Every published message is scrubbed of secret slice values (e.g. reference solutions)
    if bus is not None:
        bus = RedactingBus(bus, secret_values(slices))

    # Round 0: Parallel proposals
    proposals = await asyncio.gather(*(agent.propose(slices[agent.name]) for agent in agents))
    current_positions = list(proposals)
    rounds: list[list[Position]] = [list(current_positions)]

    if bus is not None:
        await bus.publish(
            "deliberation/round/0",
            {"round": 0, "positions": [p.to_dict() for p in current_positions]},
        )

    # If voting positions already agree at round 0, consensus is achieved immediately
    outcome, final_level, cap_applied, ask_learner = evaluate_stop_rules(rounds)
    if outcome == OutcomeKind.CONSENSUS:
        decision = Decision(
            outcome=outcome,
            final_level=final_level,
            rounds=rounds,
            cap_applied=cap_applied,
            ask_learner=ask_learner,
            rounds_used=len(rounds),
        )
        if bus is not None:
            await bus.publish("deliberation/decision", decision.to_dict())
        return decision

    # Rounds 1..max_rounds
    for r in range(1, max_rounds + 1):
        revised = await asyncio.gather(
            *(agent.revise(slices[agent.name], list(current_positions), r) for agent in agents)
        )
        revised_positions = list(revised)
        rounds.append(list(revised_positions))

        if bus is not None:
            await bus.publish(
                f"deliberation/round/{r}",
                {"round": r, "positions": [p.to_dict() for p in revised_positions]},
            )

        # Check early termination:
        # 1. Consensus reached
        round_outcome, _, _, _ = evaluate_stop_rules(rounds)
        if round_outcome == OutcomeKind.CONSENSUS:
            break

        # 2. Unanimous hold: every agent kept its previous Position
        if all(curr == prev for curr, prev in zip(revised_positions, current_positions)):
            break

        current_positions = revised_positions

    outcome, final_level, cap_applied, ask_learner = evaluate_stop_rules(rounds)
    decision = Decision(
        outcome=outcome,
        final_level=final_level,
        rounds=rounds,
        cap_applied=cap_applied,
        ask_learner=ask_learner,
        rounds_used=len(rounds),
    )

    if bus is not None:
        await bus.publish("deliberation/decision", decision.to_dict())

    return decision
