"""Sherpa's three peer agents: Learning, Momentum, and Integrity.

Each agent is a role over a shared JevClient. Code owns known facts (thresholds on
counters, hint timing) and states them as Arguments with evidence; Jev supplies the
semantic judgments. In revise(), an agent has Jev judge each peer argument that
pushes toward the peer's vote against what this agent observes. Arguments that hold
move it one level; arguments that fail become rebuttals the UI and explainer show.
"""

from __future__ import annotations

import asyncio
from typing import Any

from contracts import (
    MAX_LEVEL,
    MAX_LEVEL_STEP_PER_ROUND,
    MIN_LEVEL,
    Argument,
    HelpLevel,
    JevClient,
    Position,
    SliceDict,
    argument_id,
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

# REASON_TAG -> (direction, what the claim means). Direction: +1 more help, -1 less help.
CLAIMS: dict[str, tuple[int, str]] = {
    "concept_gap": (1, "the learner seems to be missing a programming concept, not just making a slip"),
    "repeated_error": (1, "the learner keeps hitting the same error"),
    "long_idle": (1, "the learner has stopped editing for a long time"),
    "learner_requested_help": (1, "the learner asked for help"),
    "active_progress": (-1, "the learner is making progress on their own"),
    "recent_hint_given": (-1, "the learner just got a hint and should have time to use it"),
    "solution_leak_risk": (-1, "pointing at the problem would give away the solution"),
}

# Tunable thresholds. Starting guesses; adjust them on real runs.
TAG_PROBABILITY = 0.6  # a Noul above this adds its reason tag
ACCEPT_PROBABILITY = 0.6  # a peer argument judged at or above this moves the agent
REJECT_PROBABILITY = 0.4  # at or below this, the agent rebuts it; in between it neither moves nor rebuts
RECENT_HINT_SECONDS = 120
LONG_IDLE_SECONDS = 180
REPEATED_ERROR_COUNT = 2
DEFAULT_POLICY_MAX_LEVEL = HelpLevel.POINT_TO_AREA


def _to_level(value: float) -> int:
    return min(max(round(value), MIN_LEVEL), MAX_LEVEL)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _responds_to(argument: Argument) -> str | None:
    """Id of the peer argument this one accepts or rebuts, if it is a response."""
    return argument.evidence.get("responds_to")


class BaseAgent:
    """Shared propose/revise flow. Subclasses set name and goal and implement _initial_position."""

    name: str
    goal: str
    levels: list[str] = VOTING_LEVELS  # scale this agent scores and votes on; index = level

    def __init__(self, jev: JevClient) -> None:
        self.jev = jev
        self._last: Position | None = None
        self._judged: dict[str, float] = {}  # peer argument id -> probability it holds

    async def propose(self, slice: SliceDict) -> Position:
        level, confidence, evidence, cap = await self._initial_position(slice)
        arguments = [
            Argument(
                id=argument_id(self.name, 0, claim),
                agent=self.name,
                round=0,
                claim=claim,
                direction=CLAIMS[claim][0],
                evidence=facts,
            )
            for claim, facts in evidence.items()
        ]
        self._last = Position(
            agent=self.name,
            round=0,
            level=level,
            confidence=confidence,
            reason_tags=list(evidence),
            cap=cap,
            arguments=arguments,
        )
        return self._last

    async def revise(self, slice: SliceDict, peers: list[Position], round: int) -> Position:
        previous = self._last
        if previous is None:
            raise RuntimeError(f"{self.name}: revise() called before propose()")

        # Peer arguments that would pull this agent toward that peer's vote. Integrity's
        # cap already clamps every vote, so its arguments are not re-litigated here.
        pulling = [
            argument
            for peer in peers
            if peer.agent != self.name and peer.cap is None
            for argument in peer.arguments
            if argument.attacks is None
            and _responds_to(argument) is None
            and argument.direction != 0
            and argument.direction == _sign(peer.level - previous.level)
        ]
        new = [a for a in pulling if a.id not in self._judged]
        judgments = await asyncio.gather(*(self._judge(slice, previous, a) for a in new))
        for argument, probability in zip(new, judgments):
            self._judged[argument.id] = probability

        accepted = [a for a in pulling if self._judged[a.id] >= ACCEPT_PROBABILITY]
        responses = [self._response(a, round) for a in new if not REJECT_PROBABILITY < self._judged[a.id] < ACCEPT_PROBABILITY]

        step = _sign(sum(a.direction for a in accepted)) * MAX_LEVEL_STEP_PER_ROUND
        ceiling = len(self.levels) - 1 if previous.cap is None else min(previous.cap, len(self.levels) - 1)
        if step and not MIN_LEVEL <= previous.level + step <= ceiling:
            step = 0
        if step == 0 and not responses:
            return previous

        moved_by = [self._judged[a.id] for a in accepted if a.direction == step] if step else []
        self._last = Position(
            agent=self.name,
            round=round,
            level=previous.level + step,
            confidence=sum(moved_by) / len(moved_by) if moved_by else previous.confidence,
            reason_tags=previous.reason_tags,
            cap=previous.cap,
            arguments=[*previous.arguments, *responses],
        )
        return self._last

    async def _judge(self, slice: SliceDict, mine: Position, argument: Argument) -> float:
        """Probability that a peer's argument holds given what this agent observes."""
        state = {
            **self._state(slice),
            "my_vote": {
                "level": mine.level,
                "meaning": self.levels[min(mine.level, len(self.levels) - 1)],
                "my_reasons": [CLAIMS[a.claim][1] for a in mine.arguments if a.claim in CLAIMS and _responds_to(a) is None],
            },
            "peer_argument": {
                "agent": argument.agent,
                "wants": "more help" if argument.direction > 0 else "less help",
                "claim": CLAIMS[argument.claim][1],
                "evidence": argument.evidence,
            },
        }
        question = (
            "Another agent argues the learner should get `peer_argument.wants` than `my_vote`, "
            "because `peer_argument.claim`, citing `peer_argument.evidence`. Check that claim "
            "against the learner's code, last run, and the rest of `observations`. Is the claim "
            "true here, and strong enough given your goal in `agent` to move your vote one level "
            "toward it?"
        )
        result = await self.jev.noul(state, question)
        return result.probability

    def _response(self, argument: Argument, round: int) -> Argument:
        """Record this agent's verdict on a peer argument: accepted, or a rebuttal."""
        probability = self._judged[argument.id]
        held = probability >= ACCEPT_PROBABILITY
        return Argument(
            id=f"{self.name}.r{round}.re.{argument.id}",
            agent=self.name,
            round=round,
            claim=argument.claim,
            direction=argument.direction if held else 0,
            evidence={"responds_to": argument.id, "holds": _round_probability(probability)},
            attacks=None if held else argument.id,
        )

    async def _initial_position(
        self, slice: SliceDict
    ) -> tuple[int, float, dict[str, dict[str, Any]], int | None]:
        """Return (level, confidence, {reason_tag: evidence}, cap) for the round-0 proposal."""
        raise NotImplementedError

    def _state(self, slice: SliceDict, signals: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "agent": {"name": self.name, "goal": self.goal},
            "help_levels": self.levels,
            "observations": dict(slice),
        }
        if signals:
            # Code-measured facts, spelled out so Jev does not have to guess what raw counters mean.
            state["signals"] = [f"{CLAIMS[claim][1]} ({_facts(evidence)})" for claim, evidence in signals.items()]
        return state


def _round_probability(value: float) -> float:
    return round(value, 2)


def _facts(evidence: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in evidence.items())


def _last_error(slice: SliceDict) -> str | None:
    last_run = slice.get("last_run") or {}
    return last_run.get("error")


class LearningAgent(BaseAgent):
    name = "learning"
    goal = "Protect the learner's learning: give the least help that keeps them productively struggling."

    async def _initial_position(
        self, slice: SliceDict
    ) -> tuple[int, float, dict[str, dict[str, Any]], int | None]:
        evidence: dict[str, dict[str, Any]] = {}
        recent = [h for h in slice.get("hint_history", []) if h["seconds_ago"] < RECENT_HINT_SECONDS]
        if recent:
            evidence["recent_hint_given"] = {"last_hint_level": recent[-1]["level"], "seconds_ago": recent[-1]["seconds_ago"]}
        if slice.get("recent_progress"):
            evidence["active_progress"] = {"recent_progress": True, "seconds_on_task": slice.get("seconds_on_task", 0)}

        state = self._state(slice, evidence)
        level, gap = await asyncio.gather(
            self.jev.score(
                state,
                "Read the learner's `code` and `last_run` in `observations`, along with their skill "
                "level, recent hints, and progress. How much help from `help_levels` should they get "
                "now so they keep learning rather than being handed the answer?",
                self.levels,
            ),
            self.jev.noul(
                state,
                "Looking at the learner's `code` and `last_run.error` in `observations`, is the "
                "learner missing understanding of a programming concept, rather than making a typo "
                "or small slip?",
            ),
        )
        if gap.probability > TAG_PROBABILITY:
            evidence = {
                "concept_gap": {"p_concept_gap": _round_probability(gap.probability), "last_error": _last_error(slice)},
                **evidence,
            }
        return _to_level(level.value), level.confidence, evidence, None


class MomentumAgent(BaseAgent):
    name = "momentum"
    goal = "Keep the learner moving: step in before they stall out or give up."

    async def _initial_position(
        self, slice: SliceDict
    ) -> tuple[int, float, dict[str, dict[str, Any]], int | None]:
        evidence: dict[str, dict[str, Any]] = {}
        if slice.get("repeated_error_count", 0) >= REPEATED_ERROR_COUNT:
            evidence["repeated_error"] = {
                "repeated_error_count": slice["repeated_error_count"],
                "last_error": _last_error(slice),
            }
        if slice.get("seconds_since_last_edit", 0) >= LONG_IDLE_SECONDS:
            evidence["long_idle"] = {
                "seconds_since_last_edit": slice["seconds_since_last_edit"],
                "idle_threshold_seconds": LONG_IDLE_SECONDS,
            }

        state = self._state(slice, evidence)
        level = self.jev.score(
            state,
            "Given `signals`, the idle time, repeated errors, failed runs, rewrites, and `last_run` in "
            "`observations`, how much help from `help_levels` does the learner need now to get unstuck?",
            self.levels,
        )
        if slice.get("learner_message"):
            asked = self.jev.noul(
                state, "Is `observations.learner_message` the learner asking for help or a hint?"
            )
            level_result, asked_result = await asyncio.gather(level, asked)
        else:
            level_result, asked_result = await level, None

        if asked_result is not None and asked_result.probability > TAG_PROBABILITY:
            evidence["learner_requested_help"] = {
                "learner_message": slice.get("learner_message"),
                "p_asked": _round_probability(asked_result.probability),
            }
        return _to_level(level_result.value), level_result.confidence, evidence, None


class IntegrityAgent(BaseAgent):
    """Sets a hard cap on the help level in round 0 and holds it.

    The loop reads only Integrity's cap (its level is excluded from the spread),
    so revising its level would change nothing; revise() holds without calling Jev.
    Every hint is also leak-checked against the reference solution after it is
    written (hints.write_hint_async).
    """

    name = "integrity"
    goal = "Never let a hint give away the task's solution."
    levels = LEVEL_DESCRIPTIONS  # judges the full scale; the cap is then clamped to policy_max_level

    async def revise(self, slice: SliceDict, peers: list[Position], round: int) -> Position:
        if self._last is None:
            raise RuntimeError(f"{self.name}: revise() called before propose()")
        return self._last

    async def _initial_position(
        self, slice: SliceDict
    ) -> tuple[int, float, dict[str, dict[str, Any]], int | None]:
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
        evidence: dict[str, dict[str, Any]] = {}
        if leak.probability > TAG_PROBABILITY:
            evidence["solution_leak_risk"] = {"task_id": slice.get("task_id"), "p_leak": _round_probability(leak.probability)}
            cap = min(cap, HelpLevel.CONCEPTUAL_HINT)
        return int(cap), safe.confidence, evidence, int(cap)
