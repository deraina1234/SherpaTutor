import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from contracts import JevNoul
from jev import FakeJevClient
from hints import ALL_TESTS_PASS_TEXT, HintResult
from server import _hint, create_app, nudge_level, round_message, to_proposal
from state_builder import task_for

PID = "second-largest"
SECRET = task_for(PID).reference_solution_summary
CODE = "def second_largest(nums):\n    nums.sort()\n    return nums[-2]\n"


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


def test_to_proposal_lists_responses_to_peer_arguments() -> None:
    rebuttal = {
        "id": "learning.r1.re.momentum.r0.long_idle", "agent": "learning", "round": 1, "claim": "long_idle",
        "direction": 0, "evidence": {"responds_to": "momentum.r0.long_idle", "holds": 0.2}, "attacks": "momentum.r0.long_idle",
        "text": "They ran the tests 40 seconds ago, so they are not idle.",
    }
    own = {
        "id": "learning.r0.active_progress", "agent": "learning", "round": 0, "claim": "active_progress",
        "direction": -1, "evidence": {"recent_progress": True}, "attacks": None, "text": "Two more tests pass than before.",
    }
    untexted = {**own, "id": "learning.r0.recent_hint_given", "claim": "recent_hint_given", "text": None}
    position = {
        "agent": "learning", "level": 0, "confidence": 0.5, "reason_tags": [], "cap": None,
        "arguments": [own, untexted, rebuttal],
    }
    proposal = to_proposal(position)
    assert proposal["arguments"] == [{"claim": "active_progress", "text": "Two more tests pass than before."}]
    assert proposal["responses"] == [
        {
            "to": "momentum", "claim": "long_idle", "accepted": False, "holds": 0.2,
            "text": "They ran the tests 40 seconds ago, so they are not idle.",
        }
    ]


def test_nudge_on_solved_code_says_tests_pass_instead_of_hinting(tmp_path) -> None:
    jev = FakeJevClient(score_value=1, noul_probability=0.1)
    with TestClient(create_app(jev=jev, log_path=tmp_path / "sessions.jsonl")) as c, c.websocket_connect("/ws") as ws:
        ws.send_json(event("problem", seconds_ago=60))
        ws.send_json(event("edit", seconds_ago=30, snapshot=CODE, diff_summary={"added": 2, "deleted": 1}))
        ws.send_json(event("run", seconds_ago=10, passed=True, first_error=None))
        ws.send_json(event("ask_nudge"))
        assert ws.receive_json() == {"type": "hint", "level": 0, "text": ALL_TESTS_PASS_TEXT}
    assert jev.calls == []  # no deliberation, no leak check


def test_agents_word_their_arguments_when_the_app_has_a_writer(tmp_path) -> None:
    class Jev(FakeJevClient):
        async def noul(self, state: dict, question: str):
            if "argument_text" in state and "will read" in question:
                self.calls.append(("noul", state, question))
                return JevNoul(probability=0.1, confidence=0.9)  # does not point at the fix
            return await super().noul(state, question)

    prompts: list[str] = []

    async def writer(prompt: str) -> str:
        prompts.append(prompt)
        return '{"concept_gap": "Case 2 returns 5 where 4 is expected."}'

    app = create_app(jev=Jev(score_value=1, noul_probability=0.9), log_path=tmp_path / "sessions.jsonl", writer=writer)
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.send_json(event("problem", seconds_ago=60))
        ws.send_json(event("edit", seconds_ago=30, snapshot=CODE, diff_summary={"added": 2, "deleted": 1}))
        ws.send_json(event("run", passed=False, first_error="Case 2: expected 4, got 5"))
        round0 = ws.receive_json()

    by_agent = {p["agent"]: p for p in round0["proposals"]}
    assert by_agent["learning"]["arguments"] == [{"claim": "concept_gap", "text": "Case 2 returns 5 where 4 is expected."}]
    assert "arguments" not in by_agent["integrity"]
    assert all("reference_solution_summary" not in prompt for prompt in prompts)
