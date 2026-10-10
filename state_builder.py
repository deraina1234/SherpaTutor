"""State builder: turns browser events into the three agents' private slices.

The browser streams these events over the WebSocket (see frontend/src/socket.ts).
Every one carries the problem_id of the problem on screen:

    {"type": "problem",   "ts": ms, "problem_id": str}   # learner opened / switched problem
    {"type": "edit",      "ts": ms, "problem_id": str, "snapshot": str, "diff_summary": {...}}
    {"type": "run",       "ts": ms, "problem_id": str, "passed": bool, "first_error": str | None}
    {"type": "chat",      "ts": ms, "problem_id": str, "text": str}
    {"type": "ask_nudge", "ts": ms, "problem_id": str}

StateBuilder keeps the running state, derives features in code (design doc §9: the
agents should not have to infer "rewrote the same function 3 times" from raw logs),
cuts one slice per agent matching contracts.py, and says when a deliberation should
start.

Usage from the backend (one Session per WebSocket connection):

    session = Session()

    # every WebSocket message
    trigger = session.ingest(event)
    # once a second from a background task (idle and heartbeat triggers)
    trigger = session.tick()

    if trigger:
        slices = session.slices()          # {"learning": ..., "momentum": ..., "integrity": ...}
        decision = await run_deliberation(slices, ...)
        session.record_hint(level)         # after a hint is actually shown

Session keeps one StateBuilder per problem; StateBuilder holds a single problem's state.

Time: event "ts" values are browser milliseconds (Date.now()). Everything inside is
seconds. Live, "now" is the wall clock. For replaying a recorded JSONL session, build
with Session(clock=None) or StateBuilder(task, clock=None) so "now" is the latest event time and replays are
deterministic (or pass `now=` explicitly to tick()/slices()).

Stdlib only, Python 3.11+.
"""

from __future__ import annotations

import difflib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contracts import HelpLevel, HintRecord, IntegritySlice, LearningSlice, MomentumSlice

# ---------------------------------------------------------------------------
# Tunable thresholds. Starting guesses; adjust on recorded sessions.
# LONG_IDLE_SECONDS matches agents.LONG_IDLE_SECONDS so the trigger and the tag agree.
# ---------------------------------------------------------------------------

LONG_IDLE_SECONDS = 180  # idle trigger fires once per idle stretch
HEARTBEAT_SECONDS = 20  # deliberate at least this often while the learner is active
MIN_TRIGGER_GAP_SECONDS = 3  # debounce automatic triggers (chat / ask_nudge always fire)
RECENT_PROGRESS_SECONDS = 120  # a progress run this recent counts as recent_progress
ACTIVE_EDIT_SECONDS = 30  # editing this recently also counts as recent_progress
LEARNER_MESSAGE_TTL_SECONDS = 300  # how long a chat message stays in Momentum's slice
HINT_HISTORY_LIMIT = 10
CODE_EXCERPT_LINES = 150  # design doc §9: truncate code to ~150 lines


class Trigger(str, Enum):
    """Why a deliberation should start. Log it alongside the decision."""

    FAILED_RUN = "failed_run"
    LONG_IDLE = "long_idle"
    LEARNER_MESSAGE = "learner_message"
    ASK_NUDGE = "ask_nudge"  # learner clicked "Want a nudge?" (usually after a deadlock)
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class TaskInfo:
    """Per-problem facts. Only Integrity sees reference_solution_summary."""

    task_id: str
    reference_solution_summary: str
    policy_max_level: int = HelpLevel.POINT_TO_AREA


# Keyed by the frontend problem id (frontend/src/problems.ts). Integrity is the only agent
# that sees these summaries. Problems missing here fall back to UNKNOWN_TASK.
_SOLUTIONS: dict[str, str] = {
    'second-largest': (
        'Remove duplicates (e.g. set), sort or scan once tracking the top two distinct values, and return the second one. Duplicates of the maximum must not count.'
    ),
    'two-sum': (
        'One pass with a dict from value to index: for each number, check whether target - number was already seen, and return [seen_index, i].'
    ),
    'palindrome-number': (
        'Return False for negatives; otherwise compare the number with its reversal (str(x) == str(x)[::-1], or reverse the digits arithmetically).'
    ),
    'roman-to-integer': (
        'Map symbols to values and walk left to right, subtracting a value when it is smaller than the next one and adding it otherwise.'
    ),
    'valid-parentheses': (
        'Use a stack: push opening brackets, and on a closing bracket pop and check it matches; the string is valid only if the stack ends empty.'
    ),
    'longest-common-prefix': (
        "Take the first string and shorten it (or compare column by column) until every string starts with it; return '' for an empty list."
    ),
    'remove-duplicates': (
        'Walk the sorted list and append a value only when it differs from the last value appended.'
    ),
    'search-insert-position': (
        'Binary search with lo/hi; when the loop ends without finding target, lo is the insertion index.'
    ),
    'maximum-subarray': (
        "Kadane's algorithm: keep the best sum ending here as max(x, current + x) and track the overall maximum, starting from the first element."
    ),
    'plus-one': (
        'Add one from the last digit, turning 9s into 0s and carrying; if every digit was 9, prepend a 1.'
    ),
    'climbing-stairs': (
        'Fibonacci-style DP: ways(n) = ways(n-1) + ways(n-2) with ways(1)=1, ways(2)=2, iterating with two variables.'
    ),
    'merge-sorted-lists': (
        'Two pointers, append the smaller current element each step, then append whatever remains of either list.'
    ),
    'best-time-to-buy-and-sell-stock': (
        'One pass tracking the minimum price so far and the best profit as max(best, price - min_price).'
    ),
    'valid-palindrome': (
        'Filter to lowercase alphanumeric characters and compare with the reverse (or two pointers skipping other characters).'
    ),
    'single-number': (
        'XOR all values together; pairs cancel and the single value remains (a Counter also works).'
    ),
    'contains-duplicate': (
        'Compare len(set(nums)) with len(nums), or add to a set and return True on the first repeat.'
    ),
    'valid-anagram': (
        'Compare letter counts (Counter(s) == Counter(t)) or compare sorted(s) with sorted(t).'
    ),
    'missing-number': (
        'Return n*(n+1)//2 - sum(nums) where n = len(nums) (or XOR indices with values).'
    ),
    'move-zeroes': (
        'Build the list of non-zero values in order, then append as many zeros as were removed.'
    ),
    'fizz-buzz': (
        'Loop 1..n and check divisibility by 15 first, then 3, then 5, otherwise str(i).'
    ),
    'product-of-array-except-self': (
        'Two passes: prefix products left to right into the result, then multiply by suffix products right to left, without division.'
    ),
    '3sum': (
        'Sort, fix each index i (skipping duplicate values), then two pointers on the rest moving inward by sum sign and skipping duplicates.'
    ),
    'longest-substring-without-repeating-characters': (
        'Sliding window with a dict of last seen index; move the left edge past a repeated character and track the max window length.'
    ),
    'container-with-most-water': (
        'Two pointers at both ends; record min(height) * width and move the pointer at the shorter line inward.'
    ),
    'rotate-array': (
        'Use k %= len(nums), then return nums[-k:] + nums[:-k] (handle k == 0 and empty lists).'
    ),
    'coin-change': (
        'Bottom-up DP over amounts 0..amount: dp[a] = min(dp[a - c] + 1) over coins, with infinity for unreachable and -1 at the end if still infinite.'
    ),
    'house-robber': (
        'DP with two variables: best = max(prev_best, prev_prev_best + nums[i]) for each house.'
    ),
    'longest-increasing-subsequence': (
        'O(n^2) DP where dp[i] = 1 + max dp[j] for j < i with nums[j] < nums[i], or patience sorting with bisect for O(n log n).'
    ),
    'search-in-rotated-sorted-array': (
        "Modified binary search: find which half is sorted, check whether target lies in that half's range, and discard the other half; return -1 if missing."
    ),
    'merge-intervals': (
        'Sort by start, then extend the last merged interval while the next start is <= its end (touching counts), otherwise start a new one.'
    ),
    'subarray-sum-equals-k': (
        'Prefix sums with a Counter seeded {0: 1}: for each running sum add count[running - k], then increment count[running].'
    ),
}

TASKS: dict[str, TaskInfo] = {pid: TaskInfo(pid, summary) for pid, summary in _SOLUTIONS.items()}


def task_for(problem_id: str | None) -> TaskInfo:
    """TaskInfo for a frontend problem id, with a cautious fallback for unknown ids."""
    if problem_id in TASKS:
        return TASKS[problem_id]
    return TaskInfo(
        task_id=problem_id or "unknown",
        reference_solution_summary="Unknown task: no reference solution available.",
        policy_max_level=HelpLevel.CONCEPTUAL_HINT,
    )


# ---------------------------------------------------------------------------
# Feature helpers (pure functions, unit-tested)
# ---------------------------------------------------------------------------

_CASE_PREFIX = re.compile(r"^\s*case\s+\d+\s*:\s*", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(\.\d+)?")
_DEF = re.compile(r"^(\s*)def\s+(\w+)\s*\(")
_FIRST_CASE = re.compile(r"^\s*case\s+(\d+)\s*:", re.IGNORECASE)


def normalize_error(error: str | None) -> str | None:
    """Collapse an error to its 'kind' so the same mistake on different inputs matches.

    "Case 2: expected 4, got None" and "Case 3: expected 9, got None" both become
    "expected n, got none".
    """
    if not error:
        return None
    text = _CASE_PREFIX.sub("", error.strip())
    text = _NUMBER.sub("N", text)
    return " ".join(text.lower().split())


def first_failing_case(error: str | None) -> int | None:
    """Return the case number from 'Case 3: ...', or None for whole-program errors."""
    if not error:
        return None
    m = _FIRST_CASE.match(error)
    return int(m.group(1)) if m else None


def function_bodies(code: str) -> dict[str, list[str]]:
    """Map each `def name(` to its lines (header plus indented body).

    Uses indentation rather than `ast` because the learner's code often does not
    parse mid-edit. Nested defs are folded into their parent.
    """
    lines = code.splitlines()
    bodies: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        m = _DEF.match(lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1).expandtabs())
        name = m.group(2)
        body = [lines[i]]
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= indent and not line.lstrip().startswith("#"):
                break
            body.append(line)
            i += 1
        while body and not body[-1].strip():
            body.pop()
        bodies[name] = [b.rstrip() for b in body]
    return bodies


def is_rewrite(before: list[str], after: list[str]) -> bool:
    """True if existing lines were replaced or deleted, not just appended to.

    Typing new lines into an empty function is progress-shaped; changing lines that
    were already there is what "rewriting the same function" means.
    """
    if before == after:
        return False
    ops = difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
    for tag, i1, i2, _, _ in ops:
        if tag in ("replace", "delete") and any(line.strip() for line in before[i1:i2]):
            return True
    return False


def changed_line(before: str, after: str) -> int | None:
    """Index of the first line in `after` that differs from `before` (approximate cursor)."""
    a, b = before.splitlines(), after.splitlines()
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            return min(j1, max(len(b) - 1, 0)) if b else None
    return None


# ---------------------------------------------------------------------------
# StateBuilder
# ---------------------------------------------------------------------------


@dataclass
class _Hint:
    level: int
    at: float


@dataclass
class StateBuilder:
    task: TaskInfo
    learner_level: str = "beginner"
    clock: Callable[[], float] | None = time.time  # None = replay: "now" is the latest event time
    start: float | None = None  # session start, seconds; defaults to the first event

    # Raw state
    code: str = ""
    last_edit_at: float | None = None
    last_run_at: float | None = None
    last_passed: bool | None = None
    last_error: str | None = None
    learner_message: str | None = None
    learner_message_at: float | None = None
    resumed_at: float | None = None  # last time the learner switched to this problem
    hints: list[_Hint] = field(default_factory=list)

    # Derived counters
    repeated_error_count: int = 0
    failed_runs_since_progress: int = 0
    last_progress_at: float | None = None
    rewrites: dict[str, int] = field(default_factory=dict)  # per function, since last progress
    total_runs: int = 0
    total_edits: int = 0
    _last_error_kind: str | None = None
    _best_first_fail: int | None = None  # highest "Case N" reached since the last pass
    _best_passed_count: int | None = None
    _bodies: dict[str, list[str]] = field(default_factory=dict)
    _cursor_line: int | None = None

    # Trigger bookkeeping
    _latest: float | None = None  # newest event time seen, used as "now" when none is given
    _last_trigger_at: float | None = None
    _idle_fired: bool = False
    _activity_since_trigger: bool = False

    # -- public API ---------------------------------------------------------

    def ingest(self, event: dict[str, Any], now: float | None = None) -> Trigger | None:
        """Apply one browser event. Returns a Trigger if a deliberation should start."""
        t = self._event_time(event, now)
        kind = event.get("type")
        if kind == "edit":
            self._on_edit(str(event.get("snapshot", "")), t)
            return None  # edits never trigger directly (no per-keystroke deliberation)
        if kind == "run":
            failed = self._on_run(bool(event.get("passed")), event.get("first_error"), t, event)
            return self._fire(Trigger.FAILED_RUN, t) if failed else None
        if kind == "chat":
            text = str(event.get("text", "")).strip()
            if not text:
                return None
            self.learner_message, self.learner_message_at = text, t
            self._activity_since_trigger = True
            return self._fire(Trigger.LEARNER_MESSAGE, t, force=True)
        if kind == "ask_nudge":
            return self._fire(Trigger.ASK_NUDGE, t, force=True)
        return None

    def tick(self, now: float | None = None) -> Trigger | None:
        """Check time-based triggers. Call about once a second from the backend."""
        t = self._now(now)
        if self.start is None:
            return None
        if not self._idle_fired and self.seconds_since_last_edit(t) >= LONG_IDLE_SECONDS:
            self._idle_fired = True
            return self._fire(Trigger.LONG_IDLE, t, force=True)
        since = t - (self._last_trigger_at if self._last_trigger_at is not None else self.start)
        if self._activity_since_trigger and since >= HEARTBEAT_SECONDS:
            return self._fire(Trigger.HEARTBEAT, t)
        return None

    def resume(self, now: float | None = None) -> None:
        """The learner switched (back) to this problem: restart idle and heartbeat timers."""
        t = self._now(now)
        if self.start is None:
            self.start = t
        self._latest = t if self._latest is None else max(self._latest, t)
        self.resumed_at = t
        self._idle_fired = False
        self._last_trigger_at = t
        self._activity_since_trigger = False

    def record_hint(self, level: int, now: float | None = None) -> None:
        """Call after a hint is shown, so Learning sees it in hint_history."""
        self.hints.append(_Hint(level=int(level), at=self._now(now)))
        self.hints = self.hints[-HINT_HISTORY_LIMIT:]

    def slices(self, now: float | None = None) -> dict[str, LearningSlice | MomentumSlice | IntegritySlice]:
        """The three private views, keyed by agent name (contracts.AGENT_NAMES)."""
        t = self._now(now)
        return {
            "learning": self.learning_slice(t),
            "momentum": self.momentum_slice(t),
            "integrity": self.integrity_slice(),
        }

    def learning_slice(self, now: float | None = None) -> LearningSlice:
        t = self._now(now)
        history: list[HintRecord] = [
            {"level": h.level, "seconds_ago": int(t - h.at)} for h in self.hints
        ]
        return {
            "learner_level": self.learner_level,
            "hint_history": history,
            "seconds_on_task": int(t - self.start) if self.start is not None else 0,
            "recent_progress": self.recent_progress(t),
        }

    def momentum_slice(self, now: float | None = None) -> MomentumSlice:
        t = self._now(now)
        msg = self.learner_message
        if msg is not None and self.learner_message_at is not None:
            if t - self.learner_message_at > LEARNER_MESSAGE_TTL_SECONDS:
                msg = None
        return {
            "seconds_since_last_edit": int(self.seconds_since_last_edit(t)),
            "repeated_error_count": self.repeated_error_count,
            "failed_runs_since_progress": self.failed_runs_since_progress,
            "same_function_rewrites": max(self.rewrites.values(), default=0),
            "learner_message": msg,
        }

    def integrity_slice(self) -> IntegritySlice:
        return {
            "task_id": self.task.task_id,
            "reference_solution_summary": self.task.reference_solution_summary,
            "policy_max_level": int(self.task.policy_max_level),
        }

    def hint_context(self, max_lines: int = CODE_EXCERPT_LINES) -> dict[str, Any]:
        """What the LLM hint writer needs: code (truncated around the last edit) and the error."""
        return {
            "task_id": self.task.task_id,
            "code": self.code_excerpt(max_lines),
            "last_error": self.last_error,
            "last_passed": self.last_passed,
        }

    def code_excerpt(self, max_lines: int = CODE_EXCERPT_LINES) -> str:
        lines = self.code.splitlines()
        if len(lines) <= max_lines:
            return self.code
        center = self._cursor_line if self._cursor_line is not None else len(lines) - 1
        start = max(0, min(center - max_lines // 2, len(lines) - max_lines))
        return "\n".join(lines[start : start + max_lines])

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        """JSON-safe dump for the JSONL logger: every feature plus the three slices."""
        t = self._now(now)
        return {
            "t": t,
            "task_id": self.task.task_id,
            "features": {
                "seconds_on_task": int(t - self.start) if self.start is not None else 0,
                "seconds_since_last_edit": int(self.seconds_since_last_edit(t)),
                "repeated_error_count": self.repeated_error_count,
                "failed_runs_since_progress": self.failed_runs_since_progress,
                "rewrites_per_function": dict(self.rewrites),
                "recent_progress": self.recent_progress(t),
                "last_passed": self.last_passed,
                "last_error": self.last_error,
                "total_runs": self.total_runs,
                "total_edits": self.total_edits,
                "hints_given": len(self.hints),
            },
            "slices": self.slices(t),
        }

    # -- derived values -----------------------------------------------------

    def seconds_since_last_edit(self, now: float | None = None) -> float:
        t = self._now(now)
        refs = [r for r in (self.last_edit_at, self.resumed_at, self.start) if r is not None]
        return max(0.0, t - max(refs)) if refs else 0.0

    def recent_progress(self, now: float | None = None) -> bool:
        """A progress run within RECENT_PROGRESS_SECONDS, or editing within ACTIVE_EDIT_SECONDS.

        Learning's slice has no edit timing, so this is how it sees an actively working
        learner (demo scenario 2: rewriting, but hitting the same error).
        """
        t = self._now(now)
        if self.last_progress_at is not None and t - self.last_progress_at <= RECENT_PROGRESS_SECONDS:
            return True
        return self.last_edit_at is not None and t - self.last_edit_at <= ACTIVE_EDIT_SECONDS

    # -- event handlers -----------------------------------------------------

    def _on_edit(self, snapshot: str, t: float) -> None:
        if snapshot == self.code:
            return
        if self.code:
            self._cursor_line = changed_line(self.code, snapshot)
        new_bodies = function_bodies(snapshot)
        for name, body in new_bodies.items():
            old = self._bodies.get(name)
            if old is not None and is_rewrite(old, body):
                self.rewrites[name] = self.rewrites.get(name, 0) + 1
        self._bodies = new_bodies
        self.code = snapshot
        self.last_edit_at = t
        self.total_edits += 1
        self._idle_fired = False  # re-arm the idle trigger
        self._activity_since_trigger = True

    def _on_run(self, passed: bool, error: Any, t: float, event: dict[str, Any]) -> bool:
        """Update run counters. Returns True if the run failed."""
        err = None if passed or error is None else str(error)
        self.total_runs += 1
        self.last_run_at, self.last_passed, self.last_error = t, passed, err
        self._activity_since_trigger = True

        if passed:
            self._progress(t)
            self._best_first_fail = None
            self.repeated_error_count = 0
            self._last_error_kind = None
            return False

        # Progress without passing: more cases pass than before. Uses passed_count if the
        # frontend sends it, else infers it from "Case N:" moving later.
        progressed = False
        case = first_failing_case(err)
        if case is not None:
            if self._best_first_fail is not None and case > self._best_first_fail:
                progressed = True
            self._best_first_fail = max(case, self._best_first_fail or 0)
        if isinstance(event.get("passed_count"), int):
            best = self._best_passed_count
            if best is not None and event["passed_count"] > best:
                progressed = True
            self._best_passed_count = max(event["passed_count"], best or 0)

        kind = normalize_error(err)
        if kind is not None and kind == self._last_error_kind:
            self.repeated_error_count += 1
        else:
            self.repeated_error_count = 1
        self._last_error_kind = kind

        if progressed:
            self._progress(t)
            self.failed_runs_since_progress = 1
        else:
            self.failed_runs_since_progress += 1
        return True

    def _progress(self, t: float) -> None:
        self.last_progress_at = t
        self.failed_runs_since_progress = 0
        self.rewrites.clear()

    # -- triggers and time --------------------------------------------------

    def _fire(self, trigger: Trigger, t: float, force: bool = False) -> Trigger | None:
        if (
            not force
            and self._last_trigger_at is not None
            and t - self._last_trigger_at < MIN_TRIGGER_GAP_SECONDS
        ):
            return None
        self._last_trigger_at = t
        self._activity_since_trigger = False
        return trigger

    def _event_time(self, event: dict[str, Any], now: float | None) -> float:
        ts = event.get("ts")
        t = ts / 1000.0 if isinstance(ts, (int, float)) and not isinstance(ts, bool) else self._now(now)
        if self.start is None:
            self.start = t
        self._latest = t if self._latest is None else max(self._latest, t)
        return t

    def _now(self, now: float | None) -> float:
        if now is not None:
            return now
        if self.clock is None:
            return self._latest if self._latest is not None else 0.0
        wall = self.clock()
        # Live: wall clock. If the browser clock runs ahead, never go back in time.
        return max(wall, self._latest) if self._latest is not None else wall


# ---------------------------------------------------------------------------
# Session: one learner, many problems
# ---------------------------------------------------------------------------


class Session:
    """Routes events to one StateBuilder per problem_id and tracks the active problem.

    The frontend tags every event with problem_id and sends {"type": "problem"} when
    the learner switches problems. Each problem keeps its own counters, so switching
    away and back resumes where the learner left off. Only the active problem's
    triggers fire.

        session = Session()
        trigger = session.ingest(event)        # every WebSocket message
        trigger = session.tick()               # about once a second
        if trigger:
            decision = await run_deliberation(session.slices(), ...)
    """

    def __init__(self, learner_level: str = "beginner", clock: Callable[[], float] | None = time.time) -> None:
        self.learner_level = learner_level
        self.clock = clock
        self.builders: dict[str, StateBuilder] = {}
        self.active_id: str | None = None

    @property
    def active(self) -> StateBuilder | None:
        return self.builders.get(self.active_id) if self.active_id is not None else None

    def builder(self, problem_id: str) -> StateBuilder:
        if problem_id not in self.builders:
            self.builders[problem_id] = StateBuilder(
                task_for(problem_id), learner_level=self.learner_level, clock=self.clock
            )
        return self.builders[problem_id]

    def ingest(self, event: dict[str, Any], now: float | None = None) -> Trigger | None:
        pid = event.get("problem_id") or self.active_id or "unknown"
        b = self.builder(pid)
        if event.get("type") == "problem" or pid != self.active_id:
            self.active_id = pid
            ts = event.get("ts")
            t = ts / 1000.0 if isinstance(ts, (int, float)) and not isinstance(ts, bool) else now
            b.resume(t)
            if event.get("type") == "problem":
                return None
        return b.ingest(event, now)

    def tick(self, now: float | None = None) -> Trigger | None:
        b = self.active
        return b.tick(now) if b is not None else None

    def slices(self, now: float | None = None) -> dict[str, LearningSlice | MomentumSlice | IntegritySlice]:
        return self._require().slices(now)

    def record_hint(self, level: int, now: float | None = None) -> None:
        self._require().record_hint(level, now)

    def hint_context(self, max_lines: int = CODE_EXCERPT_LINES) -> dict[str, Any]:
        return self._require().hint_context(max_lines)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        return self._require().snapshot(now)

    def _require(self) -> StateBuilder:
        b = self.active
        if b is None:
            raise RuntimeError("no active problem yet; wait for a problem/edit/run event")
        return b
