import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agntcy_slim import FanoutBus, SlimSettings, describe
from contracts import AGENT_NAMES
from deliberation import InProcBus
from jev import FakeJevClient
from server import create_app
from state_builder import task_for

OASF_DIR = Path(__file__).resolve().parent.parent / "oasf"

# Taxonomy entries checked against the OASF 1.1.0 schema server. Add new ones only after
# looking them up there: records with unknown names or ids fail directory validation.
KNOWN_SKILLS = {
    "ai_ml_engineering/agent_orchestration/multi_agent_coordination": 70901,
    "ai_ml_engineering/agent_orchestration/negotiation_resolution": 70905,
    "ai_ml_engineering/ai_safety/runtime_guardrails": 71201,
    "software_engineering/code_debugging": 606,
}
KNOWN_DOMAINS = {"education/e_learning": 601}


class RecordingPublisher:
    def __init__(self) -> None:
        self.envelopes: list[dict] = []

    async def publish(self, envelope: dict) -> None:
        self.envelopes.append(envelope)


def test_settings_off_without_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLIM_ENDPOINT", raising=False)
    assert SlimSettings.from_env() is None


def test_settings_require_secret_and_parse_observers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLIM_ENDPOINT", "127.0.0.1:46357")
    monkeypatch.delenv("SLIM_SHARED_SECRET", raising=False)
    with pytest.raises(ValueError):
        SlimSettings.from_env()
    monkeypatch.setenv("SLIM_SHARED_SECRET", "x" * 27)  # SLIM: "hmac key is too short"
    with pytest.raises(ValueError):
        SlimSettings.from_env()
    monkeypatch.setenv("SLIM_SHARED_SECRET", "s" * 40)
    monkeypatch.setenv("SLIM_OBSERVERS", "a/b/one, a/b/two,")
    assert SlimSettings.from_env().observers == ("a/b/one", "a/b/two")


def test_fanout_delivers_locally_and_remotely_with_context() -> None:
    async def run() -> tuple[dict, list[dict]]:
        local, remote = InProcBus(), RecordingPublisher()
        sub = local.subscribe("*")
        bus = FanoutBus(local, remote, {"session_id": "s1"})
        await bus.publish("deliberation/round/0", {"round": 0, "positions": []})
        return await anext(sub), remote.envelopes

    local_msg, envelopes = asyncio.run(run())
    assert local_msg == {"round": 0, "positions": []}
    assert envelopes == [
        {"topic": "deliberation/round/0", "context": {"session_id": "s1"}, "message": {"round": 0, "positions": []}}
    ]


def test_describe_formats_rounds_and_decisions() -> None:
    ctx = {"session_id": "s1", "problem_id": "two-sum", "trigger": "failed_run"}
    round_line = describe({"context": ctx, "message": {"round": 1, "positions": [
        {"agent": "learning", "level": 1}, {"agent": "integrity", "level": 2, "cap": 2}]}})
    assert round_line == "[s1 two-sum failed_run] round 1: learning=L1  integrity=L2(cap 2)"
    decision = describe({"context": ctx, "message": {"outcome": "consensus", "final_level": 2, "rounds_used": 1}})
    assert decision.endswith("decision: consensus at L2 after 1 round(s)")


def test_server_publishes_redacted_deliberation_to_slim(tmp_path) -> None:
    pid = "second-largest"
    secret = task_for(pid).reference_solution_summary
    slim = RecordingPublisher()
    jev = FakeJevClient(score_value=1, noul_probability=0.1, choice_selected="hold")
    with TestClient(create_app(jev=jev, log_path=tmp_path / "s.jsonl", slim=slim)) as c, c.websocket_connect("/ws") as ws:
        now = time.time()
        for kind, ago, extra in (
            ("problem", 60, {}),
            ("edit", 30, {"snapshot": "def second_largest(nums):\n    return 0\n", "diff_summary": {"added": 1, "deleted": 0}}),
            ("run", 0, {"passed": False, "first_error": "Case 1: expected 2, got 0"}),
        ):
            ws.send_json({"type": kind, "ts": int((now - ago) * 1000), "problem_id": pid, **extra})
        while ws.receive_json()["type"] != "decision":
            pass

    topics = [e["topic"] for e in slim.envelopes]
    assert topics[0] == "deliberation/round/0" and topics[-1] == "deliberation/decision"
    assert all(e["context"] == {"session_id": e["context"]["session_id"], "problem_id": pid, "trigger": "failed_run"} for e in slim.envelopes)
    assert secret not in json.dumps(slim.envelopes)


@pytest.mark.parametrize("agent", AGENT_NAMES)
def test_oasf_record_uses_known_taxonomy(agent: str) -> None:
    record = json.loads((OASF_DIR / f"{agent}.json").read_text())
    for key in ("name", "description", "version", "schema_version", "authors", "created_at", "skills", "domains", "locators"):
        assert record[key], key
    assert record["schema_version"] == "1.1.0"
    assert all(KNOWN_SKILLS.get(s["name"]) == s["id"] for s in record["skills"]), record["skills"]
    assert all(KNOWN_DOMAINS.get(d["name"]) == d["id"] for d in record["domains"]), record["domains"]
    assert all(loc["type"] == "source_code" and loc["urls"] for loc in record["locators"])
