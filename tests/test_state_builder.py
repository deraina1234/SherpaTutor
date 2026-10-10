import asyncio
import json

from agents import IntegrityAgent, LearningAgent, MomentumAgent
from contracts import IntegritySlice, LearningSlice, MomentumSlice, Position
from jev import FakeJevClient
from state_builder import (
    HEARTBEAT_SECONDS,
    LONG_IDLE_SECONDS,
    TASKS,
    Session,
    StateBuilder,
    Trigger,
    first_failing_case,
    function_bodies,
    is_rewrite,
    normalize_error,
)

STARTER = "def second_largest(nums):\n    # your code here\n    pass\n"
T0 = 1_000_000.0  # seconds


def ms(seconds_after_start: float) -> int:
    return int((T0 + seconds_after_start) * 1000)


def edit(at: float, code: str) -> dict:
    return {"type": "edit", "ts": ms(at), "snapshot": code, "diff_summary": {"added": 0, "deleted": 0}}


def run(at: float, passed: bool, error: str | None = None) -> dict:
    return {"type": "run", "ts": ms(at), "passed": passed, "first_error": error}


def _active(s: Session) -> StateBuilder:
    assert s.active is not None
    return s.active


def builder() -> StateBuilder:
    return StateBuilder(TASKS["second-largest"], clock=None)


# -- helpers ---------------------------------------------------------------


def test_normalize_error_ignores_case_number_and_values() -> None:
    a = normalize_error("Case 2: expected 4, got None")
    b = normalize_error("Case 3: expected 9, got None")
    assert a == b == "expected n, got none"
    assert normalize_error("IndexError: list index out of range") != a
    assert normalize_error(None) is None


def test_first_failing_case() -> None:
    assert first_failing_case("Case 3: expected 9, got None") == 3
    assert first_failing_case("SyntaxError: invalid syntax") is None


def test_function_bodies_and_rewrite() -> None:
    code = "import x\n\ndef f(a):\n    b = 1\n\n    return b\n\ndef g():\n    pass\n"
    bodies = function_bodies(code)
    assert set(bodies) == {"f", "g"}
    assert bodies["f"][-1] == "    return b"
    before = ["def f():", "    x = 1"]
    assert not is_rewrite(before, before + ["    return x"])  # appending is not a rewrite
    assert is_rewrite(before, ["def f():", "    x = 2"])  # changing a line is


# -- slices match contracts.py --------------------------------------------


def test_slices_have_exactly_the_contract_keys() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    s = b.slices()
    assert set(s) == {"learning", "momentum", "integrity"}
    assert set(s["learning"]) == set(LearningSlice.__annotations__)
    assert set(s["momentum"]) == set(MomentumSlice.__annotations__)
    assert set(s["integrity"]) == set(IntegritySlice.__annotations__)
    json.dumps(b.snapshot())  # logger-safe


def test_slices_feed_the_real_agents() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest(run(10, False, "Case 1: expected 2, got None"))
    s = b.slices()
    jev = FakeJevClient(score_value=2.0, noul_probability=0.2)
    positions = asyncio.run(
        _propose_all(jev, s)
    )
    assert [p.agent for p in positions] == ["learning", "momentum", "integrity"]


async def _propose_all(jev: FakeJevClient, s: dict) -> tuple[Position, Position, Position]:
    return await asyncio.gather(
        LearningAgent(jev).propose(s["learning"]),
        MomentumAgent(jev).propose(s["momentum"]),
        IntegrityAgent(jev).propose(s["integrity"]),
    )


# -- counters --------------------------------------------------------------


def test_repeated_error_counts_same_kind_and_resets_on_new_kind() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest(run(10, False, "Case 1: expected 2, got None"))
    b.ingest(run(20, False, "Case 1: expected 2, got None"))
    b.ingest(run(30, False, "Case 1: expected 2, got None"))
    m = b.momentum_slice()
    assert m.get("repeated_error_count") == 3
    assert m.get("failed_runs_since_progress") == 3
    b.ingest(run(40, False, "Case 1: IndexError: list index out of range"))
    assert b.momentum_slice().get("repeated_error_count") == 1


def test_later_failing_case_counts_as_progress() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest(run(10, False, "Case 1: expected 2, got None"))
    b.ingest(run(20, False, "Case 1: expected 2, got None"))
    b.ingest(run(30, False, "Case 3: expected 9, got 10"))  # cases 1-2 now pass
    assert b.momentum_slice().get("failed_runs_since_progress") == 1
    assert b.learning_slice(now=T0 + 100).get("recent_progress") is True


def test_passing_run_resets_everything() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest(run(10, False, "Case 1: expected 2, got None"))
    b.ingest(run(20, True))
    m = b.momentum_slice()
    assert (m.get("repeated_error_count"), m.get("failed_runs_since_progress")) == (0, 0)


def test_same_function_rewrites() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    v1 = "def second_largest(nums):\n    return max(nums)\n"
    v2 = "def second_largest(nums):\n    return sorted(nums)[-2]\n"
    v3 = "def second_largest(nums):\n    return sorted(nums)[1]\n"
    b.ingest(edit(5, v1))
    b.ingest(edit(10, v2))
    b.ingest(edit(15, v3))
    assert b.momentum_slice().get("same_function_rewrites") == 3


def test_idle_and_hint_history_and_message_ttl() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest({"type": "chat", "ts": ms(10), "text": "I've been stuck for 20 minutes"})
    b.record_hint(1, now=T0 + 20)
    m = b.momentum_slice(now=T0 + 60)
    assert m.get("seconds_since_last_edit") == 60
    assert m.get("learner_message") == "I've been stuck for 20 minutes"
    assert b.learning_slice(now=T0 + 60).get("hint_history") == [{"level": 1, "seconds_ago": 40}]
    assert b.momentum_slice(now=T0 + 1000).get("learner_message") is None


# -- triggers --------------------------------------------------------------


def test_edits_never_trigger_but_failed_runs_and_chat_do() -> None:
    b = builder()
    assert b.ingest(edit(0, STARTER)) is None
    assert b.ingest(run(10, True)) is None
    assert b.ingest(run(20, False, "Case 1: expected 2, got None")) is Trigger.FAILED_RUN
    assert b.ingest({"type": "chat", "ts": ms(21), "text": "help"}) is Trigger.LEARNER_MESSAGE
    assert b.ingest({"type": "ask_nudge", "ts": ms(22)}) is Trigger.ASK_NUDGE


def test_failed_runs_are_debounced() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    assert b.ingest(run(10, False, "x")) is Trigger.FAILED_RUN
    assert b.ingest(run(11, False, "x")) is None  # within MIN_TRIGGER_GAP_SECONDS


def test_idle_fires_once_and_rearms_after_edit() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    assert b.tick(now=T0 + LONG_IDLE_SECONDS) is Trigger.LONG_IDLE
    assert b.tick(now=T0 + LONG_IDLE_SECONDS + 30) is None
    b.ingest(edit(LONG_IDLE_SECONDS + 40, STARTER + "\n# hi\n"))
    assert b.tick(now=T0 + 2 * LONG_IDLE_SECONDS + 40) is Trigger.LONG_IDLE


def test_heartbeat_only_with_activity() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    assert b.tick(now=T0 + HEARTBEAT_SECONDS) is Trigger.HEARTBEAT
    assert b.tick(now=T0 + 2 * HEARTBEAT_SECONDS) is None  # nothing new since
    b.ingest(edit(2 * HEARTBEAT_SECONDS + 1, STARTER + "\n# more\n"))
    assert b.tick(now=T0 + 3 * HEARTBEAT_SECONDS) is Trigger.HEARTBEAT


# -- session: many problems ------------------------------------------------


def test_session_routes_by_problem_id_and_keeps_state() -> None:
    s = Session(clock=None)
    assert s.ingest({"type": "problem", "ts": ms(0), "problem_id": "second-largest"}) is None
    s.ingest({**edit(1, STARTER), "problem_id": "second-largest"})
    s.ingest({**run(5, False, "Case 1: expected 2, got None"), "problem_id": "second-largest"})
    s.ingest({**run(10, False, "Case 1: expected 2, got None"), "problem_id": "second-largest"})
    assert _active(s).momentum_slice().get("repeated_error_count") == 2

    s.ingest({"type": "problem", "ts": ms(20), "problem_id": "two-sum"})
    assert _active(s).integrity_slice().get("task_id") == "two-sum"
    assert _active(s).momentum_slice().get("repeated_error_count") == 0

    s.ingest({"type": "problem", "ts": ms(600), "problem_id": "second-largest"})
    m = _active(s).momentum_slice(now=T0 + 601)
    assert m.get("repeated_error_count") == 2  # state kept
    assert m.get("seconds_since_last_edit") == 1  # idle clock restarted on return
    assert s.tick(now=T0 + 601) is None  # no instant idle trigger


def test_every_frontend_problem_has_a_task_and_unknown_is_capped() -> None:
    assert len(TASKS) == 31
    s = Session(clock=None)
    s.ingest({"type": "problem", "ts": ms(0), "problem_id": "not-a-problem"})
    assert _active(s).integrity_slice().get("policy_max_level") == 2


# -- recent_progress while stuck ------------------------------------------


def test_edits_stop_counting_as_progress_once_the_same_error_repeats() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest(edit(5, "def second_largest(nums):\n    return max(nums)\n"))
    b.ingest(run(6, False, "Case 1: expected 2, got 3"))
    assert b.recent_progress() is True  # one failure: editing still looks like work
    b.ingest(edit(10, "def second_largest(nums):\n    return sorted(nums)[-1]\n"))
    b.ingest(run(11, False, "Case 1: expected 2, got 3"))
    assert b.recent_progress() is False  # same error twice: stuck, not progressing
    assert b.learning_slice().get("recent_progress") is False


def test_progress_runs_still_count_while_errors_repeat() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    b.ingest(run(5, False, "Case 1: expected 2, got None"))
    b.ingest(run(10, False, "Case 3: expected 9, got 10"))  # later case fails: progress
    b.ingest(run(15, False, "Case 3: expected 9, got 10"))
    b.ingest(run(20, False, "Case 3: expected 9, got 10"))
    assert b.momentum_slice().get("repeated_error_count") == 3
    assert b.recent_progress() is True

    b.ingest(run(25, True))  # a pass always counts
    assert b.recent_progress() is True


# -- last_hint_level ------------------------------------------------------


def test_last_hint_level() -> None:
    b = builder()
    b.ingest(edit(0, STARTER))
    assert b.last_hint_level() is None
    b.record_hint(1)
    b.record_hint(2)
    assert b.last_hint_level() == 2

    s = Session(clock=None)
    assert s.last_hint_level() is None  # no active problem yet
    s.ingest({"type": "problem", "ts": ms(0), "problem_id": "second-largest"})
    assert s.last_hint_level() is None
    s.record_hint(1)
    assert s.last_hint_level() == 1
    s.ingest({"type": "problem", "ts": ms(10), "problem_id": "two-sum"})
    assert s.last_hint_level() is None  # hints are per problem
    assert s.ingest({"type": "ask_nudge", "ts": ms(20), "problem_id": "two-sum"}) is Trigger.ASK_NUDGE


# -- snapshot keeps the solution out of logs ------------------------------


def test_snapshot_has_no_reference_solution() -> None:
    s = Session(clock=None)
    s.ingest({"type": "problem", "ts": ms(0), "problem_id": "second-largest"})
    s.ingest({**run(5, False, "Case 1: expected 2, got None"), "problem_id": "second-largest"})
    summary = TASKS["second-largest"].reference_solution_summary
    dumped = json.dumps(s.snapshot())
    assert summary not in dumped
    assert "reference_solution_summary" not in dumped
    assert s.snapshot()["slices"]["integrity"]["task_id"] == "second-largest"
    # Deliberation still gets it
    assert _active(s).integrity_slice().get("reference_solution_summary") == summary

