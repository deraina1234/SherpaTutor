import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from jev import FakeJevClient
from hints import HintResult
from server import _hint, create_app, nudge_level, round_message, to_proposal
from state_builder import task_for

PID = "second-largest"
SECRET = task_for(PID).reference_solution_summary
CODE = "def second_largest(nums):\n    nums.sort()\n    return nums[-2]\n"


@pytest.fixture(autouse=True)
def offline_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    # Template hints and explanations; never call Gemini from tests.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def event(kind: str, seconds_ago: float = 0, **fields) -> dict:
    return {"type": kind, "ts": int((time.time() - seconds_ago) * 1000), "problem_id": PID, **fields}


def client(tmp_path, **jev_kwargs) -> TestClient:
    jev = FakeJevClient(**{"score_value": 1, "noul_probability": 0.1, "choice_selected": "hold", **jev_kwargs})
    return TestClient(create_app(jev=jev, log_path=tmp_path / "sessions.jsonl"))


def test_to_proposal_renames_tags_and_keeps_cap_only_when_set() -> None:
    position = {"agent": "integrity", "round": 0, "level": 2, "confidence": 0.8, "reason_tags": ["solution_leak_risk"], "cap": 2}
    assert to_proposal(position) == {"agent": "integrity", "level": 2, "confidence": 0.8, "tags": ["solution_leak_risk"], "cap": 2}
    assert "cap" not in to_proposal({**position, "agent": "learning", "cap": None})


def test_round_message_matches_socket_ts() -> None:
    message = round_message({"round": 1, "positions": [{"agent": "learning", "level": 1, "confidence": 0.5, "reason_tags": []}]})
    assert message == {"type": "round", "round": 1, "proposals": [{"agent": "learning", "level": 1, "confidence": 0.5, "tags": []}]}


@pytest.mark.parametrize(
    ("last", "cap", "expected"),
    [(None, 3, 1), (1, 3, 2), (2, 3, 3), (3, 3, 3), (2, 2, 2), (None, 0, 0)],
)
def test_nudge_level_steps_up_and_respects_cap(last: int | None, cap: int, expected: int) -> None:
    assert nudge_level(last, cap) == expected


def test_failed_run_streams_rounds_then_decision_then_hint(tmp_path) -> None:
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        # Spaced like a real learner: a run within 3s of opening a problem is debounced.
        ws.send_json(event("problem", seconds_ago=60))
        ws.send_json(event("edit", seconds_ago=30, snapshot=CODE, diff_summary={"added": 2, "deleted": 1}))
        ws.send_json(event("run", passed=False, first_error="Case 2: expected 4, got 5"))

        round0 = ws.receive_json()
        decision = ws.receive_json()
        hint = ws.receive_json()

    assert round0["type"] == "round" and round0["round"] == 0
    assert {p["agent"] for p in round0["proposals"]} == {"learning", "momentum", "integrity"}
    assert all("tags" in p for p in round0["proposals"])
    assert decision["type"] == "decision"
    assert decision["outcome"] == "consensus" and decision["level"] == 1
    assert decision["explanation"]
    assert hint == {"type": "hint", "level": 1, "text": hint["text"]} and hint["text"]
    assert SECRET not in json.dumps([round0, decision, hint])

    record = json.loads((tmp_path / "sessions.jsonl").read_text().splitlines()[0])
    assert record["metadata"] == {"trigger": "failed_run", "hint_level": 1}
    assert SECRET not in json.dumps(record)


def test_ask_nudge_escalates_one_level_without_deliberating(tmp_path) -> None:
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        ws.send_json(event("problem"))
        levels = []
        for _ in range(4):
            ws.send_json(event("ask_nudge"))
            message = ws.receive_json()
            assert message["type"] == "hint"  # no rounds or decision first
            levels.append(message["level"])

    assert levels == [1, 2, 3, 3]  # capped at second-largest's policy_max_level


def test_malformed_messages_are_ignored(tmp_path) -> None:
    with client(tmp_path) as c, c.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        ws.send_json(["not", "an", "event"])
        ws.send_json(event("problem"))
        ws.send_json(event("ask_nudge"))
        assert ws.receive_json()["type"] == "hint"


def test_health(tmp_path) -> None:
    with client(tmp_path) as c:
        assert c.get("/health").json() == {"status": "ok"}


class SlowFakeJev(FakeJevClient):
    async def score(self, state: dict, question: str, levels: list[str]):
        await asyncio.sleep(0.3)
        return await super().score(state, question, levels)


def test_triggers_queued_behind_a_running_deliberation_coalesce(tmp_path) -> None:
    jev = SlowFakeJev(score_value=1, noul_probability=0.1, choice_selected="hold")
    with TestClient(create_app(jev=jev, log_path=tmp_path / "sessions.jsonl")) as c, c.websocket_connect("/ws") as ws:
        ws.send_json(event("problem", seconds_ago=120))
        ws.send_json(event("edit", seconds_ago=90, snapshot=CODE, diff_summary={"added": 2, "deleted": 1}))
        # The first run starts a slow deliberation; the next two queue behind it.
        for ago in (60, 30, 0):
            ws.send_json(event("run", seconds_ago=ago, passed=False, first_error="Case 2: expected 4, got 5"))

        decisions = 0
        while decisions < 2:
            decisions += ws.receive_json()["type"] == "decision"
        assert ws.receive_json()["type"] == "hint"  # the second deliberation's hint
        # Nothing else is queued, so the nudge's hint comes next, not a third deliberation.
        ws.send_json(event("ask_nudge"))
        assert ws.receive_json()["type"] == "hint"

    # The running one finishes, the stale queued one is skipped, the newest runs: 2, not 3.
    assert len((tmp_path / "sessions.jsonl").read_text().splitlines()) == 2


def test_hint_uses_level_after_jev_leak_downgrade() -> None:
    assert _hint(HintResult(text="smaller hint", level=1, verified=True, downgraded=True), 2) == ("smaller hint", 1)
    assert _hint(None, 2) == (None, 2)


def test_server_runs_jev_leak_check_on_hints(tmp_path) -> None:
    jev = FakeJevClient(score_value=1, noul_probability=0.1, choice_selected="hold")
    with TestClient(create_app(jev=jev, log_path=tmp_path / "sessions.jsonl")) as c, c.websocket_connect("/ws") as ws:
        ws.send_json(event("problem"))
        ws.send_json(event("ask_nudge"))
        assert ws.receive_json()["type"] == "hint"
    leak_checks = [state for kind, state, _ in jev.calls if kind == "noul" and "hint" in state]
    assert leak_checks and leak_checks[0]["reference_solution_summary"] == SECRET
