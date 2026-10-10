"""Sherpa's three peer agents: Learning, Momentum, and Integrity.

Each agent is a role over a shared JevClient. Code owns known facts (thresholds on
counters, hint timing); Jev supplies the semantic judgments. In revise(), an agent
shows Jev its own position and its peers' positions and asks whether their reasons
should move it one level, so concessions depend on whether the peers' reasons
actually apply to what this agent observes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from contracts import (
    MAX_LEVEL,
    MAX_LEVEL_STEP_PER_ROUND,
    MIN_LEVEL,
    HelpLevel,
    JevClient,
    Position,
    SliceDict,
)

# Index = HelpLevel. Also used as the criteria for Jev level scores, so keep them descriptive.
LEVEL_DESCRIPTIONS: list[str] = [
    "no help; let the learner keep working",
    "nudge: tell the learner to look at their error",
    "conceptual hint: explain the idea they are missing, without code",
    "point to the area or line where the problem is",
    "near-solution: show most of the fix",
]
# Learning and Momentum vote on these; near-solution is blocked by default, so it is never a vote.
VOTING_LEVELS: list[str] = LEVEL_DESCRIPTIONS[: HelpLevel.NEAR_SOLUTION]

REVISE_OPTIONS: list[str] = ["lower", "hold", "raise"]
REVISE_STEPS: dict[str, int] = {"lower": -1, "hold": 0, "raise": 1}

# Tunable thresholds. Starting guesses; adjust them on real runs.
# Move only when the chosen direction beats "hold" by more than this probability margin.
# Not Choice confidence: with three close options it stays low even when the winner is clear.
MIN_REVISE_MARGIN = 0.02
TAG_PROBABILITY = 0.6  # a Noul above this adds its reason tag
RECENT_HINT_SECONDS = 120
LONG_IDLE_SECONDS = 180
REPEATED_ERROR_COUNT = 2
DEFAULT_POLICY_MAX_LEVEL = HelpLevel.POINT_TO_AREA


def _to_level(value: float) -> int:
    return min(max(round(value), MIN_LEVEL), MAX_LEVEL)


class BaseAgent:
    """Shared propose/revise flow. Subclasses set name and goal and implement _initial_position."""

    name: str
    goal: str
    levels: list[str] = VOTING_LEVELS  # scale this agent scores and votes on; index = level

    def __init__(self, jev: JevClient) -> None:
        self.jev = jev
        self._last: Position | None = None

    async def propose(self, slice: SliceDict) -> Position:
        level, confidence, tags, cap = await self._initial_position(slice)
        self._last = Position(
            agent=self.name, round=0, level=level, confidence=confidence, reason_tags=tags, cap=cap
        )
        return self._last

    async def revise(self, slice: SliceDict, peers: list[Position], round: int) -> Position:
        previous = self._last
        if previous is None:
            raise RuntimeError(f"{self.name}: revise() called before propose()")

        state = {
            "agent": {"name": self.name, "goal": self.goal},
            "help_levels": self.levels,
            "observations": dict(slice),
            "my_position": previous.to_dict(),
            "peer_positions": [p.to_dict() for p in peers if p.agent != self.name],
        }
        question = (
            "You are the agent described in `agent`, deciding how much help a programming "
            "learner should get, on the scale in `help_levels`. Your current vote is "
            "`my_position`. Other agents voted as shown in `peer_positions`, with their "
            "reasons in `reason_tags`. Considering your goal and what you see in "
            "`observations`, should you lower your vote by one level, hold it, or raise it "
            "by one level? Move only if a peer's reasons genuinely apply to what you "
            "observe and outweigh your own goal; otherwise hold."
        )
        result = await self.jev.choice(state, question, REVISE_OPTIONS)

        step = REVISE_STEPS.get(result.selected, 0) * MAX_LEVEL_STEP_PER_ROUND
        new_level = previous.level + step
        ceiling = len(self.levels) - 1 if previous.cap is None else min(previous.cap, len(self.levels) - 1)
        selected_probability = result.probabilities.get(result.selected, 0.0)
        margin = selected_probability - result.probabilities.get("hold", 0.0)
        if step == 0 or margin <= MIN_REVISE_MARGIN or not MIN_LEVEL <= new_level <= ceiling:
            return previous

        self._last = Position(
            agent=self.name,
            round=round,
            level=new_level,
            confidence=selected_probability,
            reason_tags=previous.reason_tags,
            cap=previous.cap,
        )
        return self._last

    async def _initial_position(self, slice: SliceDict) -> tuple[int, float, list[str], int | None]:
        """Return (level, confidence, reason_tags, cap) for the round-0 proposal."""
        raise NotImplementedError

    def _state(self, slice: SliceDict) -> dict[str, Any]:
        return {"agent": {"name": self.name, "goal": self.goal}, "help_levels": self.levels, "observations": dict(slice)}


class LearningAgent(BaseAgent):
    name = "learning"
    goal = "Protect the learner's learning: give the least help that keeps them productively struggling."

    async def _initial_position(self, slice: SliceDict) -> tuple[int, float, list[str], int | None]:
        state = self._state(slice)
        level, gap = await asyncio.gather(
            self.jev.score(
                state,
                "Given the learner's skill level, recent hints, and progress in `observations`, "
                "how much help from `help_levels` should they get now so they keep learning "
                "rather than being handed the answer?",
                self.levels,
            ),
            self.jev.noul(
                state,
                "Does the learner appear to be missing understanding of a programming concept, "
                "rather than making a typo or small slip?",
            ),
        )

        tags: list[str] = []
        if gap.probability > TAG_PROBABILITY:
            tags.append("concept_gap")
        if any(h["seconds_ago"] < RECENT_HINT_SECONDS for h in slice.get("hint_history", [])):
            tags.append("recent_hint_given")
        if slice.get("recent_progress"):
            tags.append("active_progress")
        return _to_level(level.value), level.confidence, tags, None


class MomentumAgent(BaseAgent):
    name = "momentum"
    goal = "Keep the learner moving: step in before they stall out or give up."

    async def _initial_position(self, slice: SliceDict) -> tuple[int, float, list[str], int | None]:
        state = self._state(slice)
        level = self.jev.score(
            state,
            "Given the idle time, repeated errors, failed runs, and rewrites in `observations`, "
            "how much help from `help_levels` does the learner need now to get unstuck?",
            self.levels,
        )
        if slice.get("learner_message"):
            asked = self.jev.noul(
                state, "Is `observations.learner_message` the learner asking for help or a hint?"
            )
            level_result, asked_result = await asyncio.gather(level, asked)
        else:
            level_result, asked_result = await level, None

        tags: list[str] = []
        if slice.get("repeated_error_count", 0) >= REPEATED_ERROR_COUNT:
            tags.append("repeated_error")
        if slice.get("seconds_since_last_edit", 0) >= LONG_IDLE_SECONDS:
            tags.append("long_idle")
        if asked_result is not None and asked_result.probability > TAG_PROBABILITY:
            tags.append("learner_requested_help")
        return _to_level(level_result.value), level_result.confidence, tags, None


class IntegrityAgent(BaseAgent):
    """Sets a hard cap on the help level in round 0 and holds it.

    The loop reads only Integrity's cap (its level is excluded from the spread),
    so revising its level would change nothing; revise() holds without calling Jev.
    """

    name = "integrity"
    goal = "Never let a hint give away the task's solution."
    levels = LEVEL_DESCRIPTIONS  # judges the full scale; the cap is then clamped to policy_max_level

    async def revise(self, slice: SliceDict, peers: list[Position], round: int) -> Position:
        if self._last is None:
            raise RuntimeError(f"{self.name}: revise() called before propose()")
        return self._last

    async def _initial_position(self, slice: SliceDict) -> tuple[int, float, list[str], int | None]:
        state = self._state(slice)
        safe, leak = await asyncio.gather(
            self.jev.score(
                state,
                "For the task in `observations`, what is the most help from `help_levels` that "
                "can be given without revealing `observations.reference_solution_summary`?",
                self.levels,
            ),
            self.jev.noul(
                state,
                "Would pointing the learner to the specific area or line of the problem "
                "effectively give away `observations.reference_solution_summary`?",
            ),
        )

        cap = min(_to_level(safe.value), slice.get("policy_max_level", DEFAULT_POLICY_MAX_LEVEL))
        tags: list[str] = []
        if leak.probability > TAG_PROBABILITY:
            tags.append("solution_leak_risk")
            cap = min(cap, HelpLevel.CONCEPTUAL_HINT)
        return int(cap), safe.confidence, tags, int(cap)
