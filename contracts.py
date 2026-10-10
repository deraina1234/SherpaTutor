"""Shared contract for Sherpa's agents, deliberation loop, bus, and Jev client.

Two people build against this file in parallel: one writes the agents (which call
Jev), the other writes the deliberation loop, bus, and stop rules. Keep it small,
typed, and stdlib-only (Python 3.11+). Change it only after agreeing out loud.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Protocol, TypedDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class HelpLevel(IntEnum):
    """How much help to give the learner. NEAR_SOLUTION is blocked by default."""

    NONE = 0
    NUDGE = 1  # "look at your error"
    CONCEPTUAL_HINT = 2
    POINT_TO_AREA = 3  # point to the area or line
    NEAR_SOLUTION = 4  # blocked by default


HELP_LEVELS: tuple[HelpLevel, ...] = tuple(HelpLevel)
MIN_LEVEL: int = HelpLevel.NONE
MAX_LEVEL: int = HelpLevel.NEAR_SOLUTION

MAX_ROUNDS: int = 3
MAX_LEVEL_STEP_PER_ROUND: int = 1

AGENT_NAMES: tuple[str, ...] = ("learning", "momentum", "integrity")

REASON_TAGS: tuple[str, ...] = (
    "repeated_error",
    "long_idle",
    "concept_gap",
    "solution_leak_risk",
    "active_progress",
    "learner_requested_help",
    "recent_hint_given",
)


def _is_valid_level(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and MIN_LEVEL <= value <= MAX_LEVEL


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    """One agent's stance on the help level for a single round."""

    agent: str
    round: int  # 0 = initial proposal
    level: int  # 0-4, see HelpLevel
    confidence: float  # 0.0-1.0
    reason_tags: list[str] = field(default_factory=list)
    cap: int | None = None  # only Integrity sets this; hard ceiling on the help level

    def __post_init__(self) -> None:
        if not isinstance(self.round, int) or isinstance(self.round, bool) or self.round < 0:
            raise ValueError(f"round must be a non-negative int, got {self.round!r}")
        if not _is_valid_level(self.level):
            raise ValueError(f"level must be an int in {MIN_LEVEL}..{MAX_LEVEL}, got {self.level!r}")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(f"confidence must be in 0.0..1.0, got {self.confidence!r}")
        unknown = [tag for tag in self.reason_tags if tag not in REASON_TAGS]
        if unknown:
            raise ValueError(f"unknown reason tags {unknown}; allowed: {REASON_TAGS}")
        if self.cap is not None and not _is_valid_level(self.cap):
            raise ValueError(f"cap must be None or an int in {MIN_LEVEL}..{MAX_LEVEL}, got {self.cap!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for JSONL logging."""
        data = asdict(self)
        data["level"] = int(self.level)
        data["cap"] = None if self.cap is None else int(self.cap)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        """Rebuild a Position from to_dict() output."""
        return cls(
            agent=data["agent"],
            round=data["round"],
            level=data["level"],
            confidence=data["confidence"],
            reason_tags=list(data.get("reason_tags", [])),
            cap=data.get("cap"),
        )


# ---------------------------------------------------------------------------
# State slices: each agent's private view.
# AGREE ON CHANGES OUT LOUD before adding, renaming, or removing fields.
# ---------------------------------------------------------------------------


class HintRecord(TypedDict):
    level: int
    seconds_ago: int


class LearningSlice(TypedDict, total=False):
    learner_level: str
    hint_history: list[HintRecord]
    seconds_on_task: int
    recent_progress: bool


class MomentumSlice(TypedDict, total=False):
    seconds_since_last_edit: int
    repeated_error_count: int
    failed_runs_since_progress: int
    same_function_rewrites: int
    learner_message: str | None


class IntegritySlice(TypedDict, total=False):
    task_id: str
    reference_solution_summary: str
    policy_max_level: int


SliceDict = LearningSlice | MomentumSlice | IntegritySlice


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


class Agent(Protocol):
    """A peer agent that proposes and revises a help level.

    Methods are async because agents call Jev over the network and the loop runs
    all three in parallel with asyncio.gather.
    """

    name: str

    async def propose(self, slice: SliceDict) -> Position:
        """Return the agent's initial Position (round 0)."""
        ...

    async def revise(self, slice: SliceDict, peers: list[Position], round: int) -> Position:
        """Return the agent's Position for `round` after seeing its peers' positions.

        Must move at most MAX_LEVEL_STEP_PER_ROUND levels from the agent's previous
        position. If the agent decides to hold, it must return that previous
        position unchanged.
        """
        ...


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class OutcomeKind(str, Enum):
    CONSENSUS = "consensus"
    SOFT_DISAGREEMENT = "soft_disagreement"
    DEADLOCK = "deadlock"


@dataclass
class Decision:
    """Result of a deliberation."""

    outcome: OutcomeKind
    final_level: int | None  # None for DEADLOCK
    rounds: list[list[Position]]  # positions per round; index = round
    cap_applied: bool
    ask_learner: bool  # True only for DEADLOCK
    rounds_used: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for JSONL logging."""
        return {
            "outcome": self.outcome.value,
            "final_level": None if self.final_level is None else int(self.final_level),
            "rounds": [[p.to_dict() for p in positions] for positions in self.rounds],
            "cap_applied": self.cap_applied,
            "ask_learner": self.ask_learner,
            "rounds_used": self.rounds_used,
        }


# ---------------------------------------------------------------------------
# Bus protocol
# ---------------------------------------------------------------------------


class Bus(Protocol):
    """Pub/sub channel between agents and the loop.

    The default implementation is an in-process async bus; a SLIM group channel
    may replace it later without other code changing.
    """

    async def publish(self, topic: str, msg: dict) -> None: ...

    def subscribe(self, topic: str) -> AsyncIterator[dict]: ...


# ---------------------------------------------------------------------------
# Jev client protocol
# NOTE: the real Jev request/response format is unconfirmed. These shapes are our
# internal contract; the real client adapts to them, and a fake can be swapped in.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JevScore:
    value: float
    probabilities: dict[int, float]
    confidence: float


@dataclass(frozen=True)
class JevNoul:
    probability: float
    confidence: float


@dataclass(frozen=True)
class JevChoice:
    selected: str
    probabilities: dict[str, float]
    confidence: float


class JevClient(Protocol):
    """The interface agents code against for Jev judgments."""

    async def score(self, state: dict, question: str, scale: tuple[int, int]) -> JevScore: ...

    async def noul(self, state: dict, question: str) -> JevNoul: ...

    async def choice(self, state: dict, question: str, options: list[str]) -> JevChoice: ...


if __name__ == "__main__":
    import json

    position = Position(
        agent="integrity",
        round=0,
        level=HelpLevel.CONCEPTUAL_HINT,
        confidence=0.8,
        reason_tags=["solution_leak_risk"],
        cap=HelpLevel.POINT_TO_AREA,
    )
    decision = Decision(
        outcome=OutcomeKind.CONSENSUS,
        final_level=HelpLevel.CONCEPTUAL_HINT,
        rounds=[[position]],
        cap_applied=False,
        ask_learner=False,
        rounds_used=1,
    )
    print(json.dumps(position.to_dict()))
    print(json.dumps(decision.to_dict()))
