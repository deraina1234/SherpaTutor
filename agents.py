"""Sherpa's three peer agents: Learning, Momentum, and Integrity.

Each agent is a role over a shared JevClient. Code owns known facts (thresholds on
counters, hint timing) and states them as Arguments with evidence; Jev supplies the
semantic judgments. In revise(), an agent has Jev judge each peer argument that
pushes toward the peer's vote against what this agent observes. Arguments that hold
move it one level; arguments that fail become rebuttals the UI and explainer show.

With a writer (an LLM call), Learning and Momentum also put each argument and
rebuttal in their own words, from their own slice. Jev checks every draft against
that slice before it is used, and peers judge the written argument, not just its
tag. Integrity never writes text: its slice holds the reference solution.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from collections.abc import Awaitable, Callable
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

log = logging.getLogger("sherpa.agents")

# Prompt -> text from an LLM, or None when no model answered.
Writer = Callable[[str], Awaitable[str | None]]

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
GROUNDED_PROBABILITY = 0.6  # a written argument must be judged at least this well supported to be used
REVEALS_PROBABILITY = 0.4  # a written argument judged above this to point at the fix is dropped
MAX_ARGUMENT_CHARS = 300  # longer drafts are dropped, not cut mid-sentence
MAX_ECHO_SIMILARITY = 0.75  # a reply this close to the argument it answers just repeats it

_GROUNDED_QUESTION = (
    "Is every factual statement in `argument_text` supported by `observations`, `signals`, or "
    "`replying_to`? Answer no if it mentions a bug, value, line, or event that is not shown there."
)
# The learner reads every argument in the agent panel, so an argument must not do a hint's job.
_REVEALS_QUESTION = (
    "The learner will read `argument_text`. Does it tell them what is wrong in their code or what "
    "to change, for example by naming a missing check, step, or condition? Reporting which test "
    "failed and what the code returned is fine."
)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _to_level(value: float) -> int:
    return min(max(round(value), MIN_LEVEL), MAX_LEVEL)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _responds_to(argument: Argument) -> str | None:
    """Id of the peer argument this one accepts or rebuts, if it is a response."""
    return argument.evidence.get("responds_to")


def _claim_text(argument: Argument) -> str:
    """What the argument says: the agent's own words if it wrote any, else the tag's meaning."""
    return argument.text or CLAIMS[argument.claim][1]


class BaseAgent:
    """Shared propose/revise flow. Subclasses set name and goal and implement _initial_position."""

    name: str
    goal: str
    levels: list[str] = VOTING_LEVELS  # scale this agent scores and votes on; index = level
    writes_arguments = True  # False for agents whose slice must never reach an LLM

    def __init__(self, jev: JevClient, writer: Writer | None = None) -> None:
        self.jev = jev
        self.writer = writer if self.writes_arguments else None
        self._last: Position | None = None
        self._signals: dict[str, dict[str, Any]] = {}  # this agent's round-0 claims and their evidence
        self._judged: dict[str, float] = {}  # peer argument id -> probability it holds

    async def propose(self, slice: SliceDict) -> Position:
        level, confidence, evidence, cap = await self._initial_position(slice)
        self._signals = evidence
        texts = await self._write(
            slice,
            level,
            {
                claim: f"Argue for {'more' if CLAIMS[claim][0] > 0 else 'less'} help because {CLAIMS[claim][1]} ({_facts(facts)})."
                for claim, facts in evidence.items()
            },
        )
        arguments = [
            Argument(
                id=argument_id(self.name, 0, claim),
                agent=self.name,
                round=0,
                claim=claim,
                direction=CLAIMS[claim][0],
                evidence=facts,
                text=texts.get(claim),
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
        decided = [a for a in new if not REJECT_PROBABILITY < self._judged[a.id] < ACCEPT_PROBABILITY]
        texts = await self._write(
            slice,
            previous.level,
            {a.id: self._reply_task(a) for a in decided},
            {a.id: {"agent": a.agent, "claim": _claim_text(a), "evidence": a.evidence} for a in decided},
        )
        responses = [self._response(a, round, texts.get(a.id)) for a in decided]

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
                "my_reasons": [_claim_text(a) for a in mine.arguments if _responds_to(a) is None],
            },
            "peer_argument": {
                "agent": argument.agent,
                "wants": "more help" if argument.direction > 0 else "less help",
                "claim": _claim_text(argument),
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

    def _reply_task(self, argument: Argument) -> str:
        probability = self._judged[argument.id]
        held = probability >= ACCEPT_PROBABILITY
        peer = argument.agent.title()
        return (
            f"{peer} argued for {'more' if argument.direction > 0 else 'less'} help: "
            f'"{_claim_text(argument)}" ({_facts(argument.evidence)}). You judged that this '
            f"{'holds' if held else 'does not hold'} ({round(probability * 100)}% likely to hold). "
            f"Start with \"{'I agree' if held else 'I disagree'}\" and give {peer} your own reason, from your goal "
            "and what you observe. Treat their facts as true: argue about what those facts mean for how much "
            "help to give, not whether they happened. Do not restate their argument."
        )

    async def _write(
        self,
        slice: SliceDict,
        level: int,
        tasks: dict[str, str],
        replying_to: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        """Have the writer word each task in this agent's voice; keep only drafts Jev finds grounded.

        `replying_to` maps a task key to the peer argument it answers, so a reply may cite it.
        """
        if self.writer is None or not tasks:
            return {}
        try:
            raw = await self.writer(self._write_prompt(slice, level, tasks))
        except Exception as err:
            log.warning("%s: writer failed (%s: %s); using tag meanings.", self.name, type(err).__name__, err)
            return {}
        drafts = _parse_drafts(raw, tasks)
        state = self._state(slice, self._signals)
        states = [{**state, "replying_to": (replying_to or {}).get(key), "argument_text": text} for key, text in drafts.items()]
        grounded, reveals = await asyncio.gather(
            asyncio.gather(*(self.jev.noul(s, _GROUNDED_QUESTION) for s in states)),
            asyncio.gather(*(self.jev.noul(s, _REVEALS_QUESTION) for s in states)),
        )
        kept: dict[str, str] = {}
        for (key, text), check, reveal in zip(drafts.items(), grounded, reveals):
            if reveal.probability > REVEALS_PROBABILITY:
                log.info("%s dropped a draft for %s that points at the fix (%.2f): %s", self.name, key, reveal.probability, text)
                continue
            answered = (replying_to or {}).get(key)
            if answered and difflib.SequenceMatcher(None, text.lower(), answered["claim"].lower()).ratio() > MAX_ECHO_SIMILARITY:
                log.info("%s dropped a reply to %s that repeats it: %s", self.name, key, text)
                continue
            if check.probability >= GROUNDED_PROBABILITY:
                kept[key] = text
            else:
                log.info("%s dropped an ungrounded draft for %s (%.2f): %s", self.name, key, check.probability, text)
        log.info("%s wrote %d of %d argument(s); %d kept.", self.name, len(drafts), len(tasks), len(kept))
        return kept

    def _write_prompt(self, slice: SliceDict, level: int, tasks: dict[str, str]) -> str:
        signals = [f"{CLAIMS[claim][1]} ({_facts(evidence)})" for claim, evidence in self._signals.items()]
        items = "\n".join(f"- {json.dumps(key)}: {task}" for key, task in tasks.items())
        return (
            f"You are the {self.name.title()} agent in Sherpa, a coding tutor where agents debate how much "
            f"help a learner should get.\nYour goal: {self.goal}\n"
            f"Your vote: level {level}, {self.levels[min(level, len(self.levels) - 1)]}.\n"
            f"What you observe (JSON): {json.dumps(dict(slice), default=str)}\n"
            f"Signals measured from that: {'; '.join(signals) or 'none'}\n\n"
            "For each item below, write one sentence of at most 35 words, addressed to the other agents. "
            "The learner reads these too, so describe their situation and behavior, never what is wrong "
            "in their code or what to change. "
            "Base it only on what you observe: cite concrete test results, counts, or times. "
            "Never invent a bug, line, value, or event. Never say how to fix the code. "
            "Write plain language a teacher would use: no field names (like p_concept_gap or "
            "seconds_since_last_edit) and no probabilities. In a test result, \"expected X, got Y\" "
            "means the learner's code returned Y where it should have returned X.\n"
            f"Items:\n{items}\n\n"
            "Reply with only a JSON object that maps each item key to its sentence."
        )

    def _response(self, argument: Argument, round: int, text: str | None = None) -> Argument:
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
            text=text,
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


def _parse_drafts(raw: str | None, tasks: dict[str, str]) -> dict[str, str]:
    """Pull {task key: sentence} out of the writer's reply, dropping anything malformed."""
    match = _JSON_OBJECT.search(raw or "")
    if match is None:
        return {}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    drafts: dict[str, str] = {}
    for key in tasks:
        value = data.get(key)
        if isinstance(value, str) and value.strip() and len(value.strip()) <= MAX_ARGUMENT_CHARS:
            drafts[key] = value.strip()
    return drafts


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
    writes_arguments = False  # its slice holds the reference solution

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
