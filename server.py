"""Sherpa backend: WebSocket server speaking the protocol in frontend/src/socket.ts.

Drop-in replacement for frontend/mock/server.js at ws://localhost:8000/ws.

    uvicorn server:app --port 8000      # or: python server.py

Per connection: browser events go into a state_builder.Session. When it fires a
trigger, the three agents deliberate (real Jev), each round streams to the browser
as it happens, then the decision (with an explanation) and, at level > 0, a hint.
"Want a nudge?" (ask_nudge) skips deliberation and gives one level more than the
last hint, never above Integrity's cap.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agents import IntegrityAgent, LearningAgent, MomentumAgent
from agntcy_slim import FanoutBus, Publisher, start_publisher_from_env
from contracts import HelpLevel, JevClient
from deliberation import InProcBus, run_deliberation
from hints import ALL_TESTS_PASS_TEXT, explain_deliberation_async, write_hint_async
from jev import TypeSafeJevClient, _load_env_file
from logger import log_deliberation
from redaction import redact, secret_values
from state_builder import Session, Trigger, task_for

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = ROOT / "logs" / "sessions.jsonl"
TICK_SECONDS = 1.0
# Background check-ins only reach the browser when they decide to help.
SILENT_TRIGGERS = {Trigger.HEARTBEAT}

log = logging.getLogger("sherpa.server")


# ---------------------------------------------------------------------------
# Protocol translation (internal contracts -> frontend/src/socket.ts InMsg)
# ---------------------------------------------------------------------------


def to_proposal(position: dict[str, Any]) -> dict[str, Any]:
    """contracts.Position.to_dict() -> socket.ts Proposal."""
    proposal = {
        "agent": position["agent"],
        "level": position["level"],
        "confidence": position["confidence"],
        "tags": position["reason_tags"],
    }
    if position.get("cap") is not None:
        proposal["cap"] = position["cap"]
    responses = [
        {
            "to": str(argument["evidence"]["responds_to"]).split(".", 1)[0],
            "claim": argument["claim"],
            "accepted": argument.get("attacks") is None,
            "holds": argument["evidence"].get("holds"),
        }
        for argument in position.get("arguments", [])
        if "responds_to" in argument.get("evidence", {})
    ]
    if responses:
        proposal["responses"] = responses
    return proposal


def round_message(bus_message: dict[str, Any]) -> dict[str, Any]:
    """A deliberation/round/N bus message -> socket.ts {type: "round"}."""
    return {
        "type": "round",
        "round": bus_message["round"],
        "proposals": [to_proposal(p) for p in bus_message["positions"]],
    }


def nudge_level(last_hint_level: int | None, cap: int) -> int:
    """One level above the last hint (at least a nudge), never above the cap."""
    return min(max((last_hint_level or 0) + 1, HelpLevel.NUDGE), cap)


def _hint(result: Any, requested_level: int) -> tuple[str | None, int]:
    """(text, level actually given) from write_hint_async's HintResult.

    Jev's leak check may downgrade a hint below the requested level.
    """
    if result is None or isinstance(result, str):
        return result, requested_level
    return getattr(result, "text", None), getattr(result, "level", requested_level)


# ---------------------------------------------------------------------------
# One browser connection
# ---------------------------------------------------------------------------


class Connection:
    def __init__(self, ws: WebSocket, jev: JevClient, log_path: Path, slim: Publisher | None = None) -> None:
        self.ws = ws
        self.jev = jev
        self.log_path = log_path
        self.slim = slim  # AGNTCY SLIM channel that also receives every (redacted) round and decision
        self.session = Session()
        self.session_id = uuid.uuid4().hex[:8]
        self._work_lock = asyncio.Lock()  # one deliberation or nudge at a time
        self._send_lock = asyncio.Lock()
        self._caps: dict[str, int] = {}  # problem_id -> Integrity's cap from its latest deliberation
        self._tasks: set[asyncio.Task[None]] = set()
        self._latest_deliberation = 0  # sequence number of the newest queued deliberation

    async def run(self) -> None:
        log.info("[%s] connected", self.session_id)
        ticker = asyncio.create_task(self._tick_loop())
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning("[%s] ignoring malformed message", self.session_id)
                    continue
                if isinstance(event, dict):
                    self._handle_trigger(self.session.ingest(event))
        except WebSocketDisconnect:
            log.info("[%s] disconnected", self.session_id)
        finally:
            ticker.cancel()
            for task in list(self._tasks):
                task.cancel()

    async def send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_json(message)

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECONDS)
            self._handle_trigger(self.session.tick())

    def _handle_trigger(self, trigger: Trigger | None) -> None:
        if trigger is None or self.session.active_id is None:
            return
        seq = 0
        if trigger is not Trigger.ASK_NUDGE:
            self._latest_deliberation += 1
            seq = self._latest_deliberation
        self._spawn(self._respond(trigger, self.session.active_id, seq))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _respond(self, trigger: Trigger, problem_id: str, seq: int) -> None:
        async with self._work_lock:
            if self.session.active_id != problem_id:
                return  # learner switched problems while this waited
            if trigger is not Trigger.ASK_NUDGE and seq < self._latest_deliberation:
                return  # a newer trigger is queued; it will deliberate on fresher state
            if self.session.solved() and trigger is not Trigger.LEARNER_MESSAGE:
                # Every test passes and nothing changed since: no hint to give. A nudge request
                # gets a plain "all tests pass" instead of a hint about a bug that is not there.
                if trigger is Trigger.ASK_NUDGE:
                    await self.send({"type": "hint", "level": 0, "text": ALL_TESTS_PASS_TEXT})
                return
            try:
                if trigger is Trigger.ASK_NUDGE:
                    await self._nudge(problem_id)
                else:
                    await self._deliberate(trigger, problem_id)
            except (WebSocketDisconnect, asyncio.CancelledError):
                raise
            except Exception:
                log.exception("[%s] %s response failed", self.session_id, trigger.value)

    async def _deliberate(self, trigger: Trigger, problem_id: str) -> None:
        slices = self.session.slices()
        secrets = secret_values(slices)
        silent = trigger in SILENT_TRIGGERS
        held_rounds: list[dict[str, Any]] = []

        local_bus = InProcBus()
        subscription = local_bus.subscribe("*")
        bus = (
            FanoutBus(local_bus, self.slim, {"session_id": self.session_id, "problem_id": problem_id, "trigger": trigger.value})
            if self.slim is not None
            else local_bus
        )

        async def forward_rounds(messages: AsyncIterator[dict[str, Any]]) -> None:
            async for message in messages:
                if "positions" not in message:
                    continue  # the decision is sent below, with its explanation
                if silent:
                    held_rounds.append(round_message(message))
                else:
                    await self.send(round_message(message))

        forwarder = asyncio.create_task(forward_rounds(subscription))
        try:
            decision = await run_deliberation(
                [LearningAgent(self.jev), MomentumAgent(self.jev), IntegrityAgent(self.jev)], slices, bus=bus
            )
        finally:
            subscription.unsubscribe()  # forwarder drains what was published, then stops
            await forwarder

        for position in decision.rounds[-1]:
            if position.cap is not None:
                self._caps[problem_id] = position.cap
        level = decision.final_level or 0
        log.info(
            "[%s] %s on %s -> %s at L%d in %d round(s)",
            self.session_id, trigger.value, problem_id, decision.outcome.value, level, decision.rounds_used,
        )

        if silent and level == 0:
            self._log(slices, decision, trigger, hint_level=None)
            return
        for message in held_rounds:
            await self.send(message)

        context = self.session.hint_context()
        explanation, hint_result = await asyncio.gather(
            explain_deliberation_async(decision, slices),
            write_hint_async(
                context["code"],
                context["last_error"],
                level,
                task_for(problem_id).reference_solution_summary,
                jev=self.jev,
                last_passed=context["last_passed"],
            )
            if level > 0
            else _none(),
        )
        hint, hint_level = _hint(hint_result, level)

        await self.send(
            {
                "type": "decision",
                "outcome": decision.outcome.value,
                "level": level,  # 0 for deadlock: the UI shows "No automatic hint" and "Want a nudge?"
                "explanation": redact(explanation, secrets),
            }
        )
        if hint:
            await self.send({"type": "hint", "level": hint_level, "text": redact(hint, secrets)})
            self.session.record_hint(hint_level)
        self._log(slices, decision, trigger, hint_level=hint_level if hint else None)

    async def _nudge(self, problem_id: str) -> None:
        last = self.session.last_hint_level()
        task = task_for(problem_id)
        cap = self._caps.get(problem_id, int(task.policy_max_level))
        level = nudge_level(last, cap)
        log.info("[%s] ask_nudge on %s: last hint L%s, cap L%d -> L%d", self.session_id, problem_id, last, cap, level)
        if level <= 0:
            return

        context = self.session.hint_context()
        hint, hint_level = _hint(
            await write_hint_async(
                context["code"],
                context["last_error"],
                level,
                task.reference_solution_summary,
                jev=self.jev,
                last_passed=context["last_passed"],
            ),
            level,
        )
        if hint:
            secrets = secret_values(self.session.slices())
            await self.send({"type": "hint", "level": hint_level, "text": redact(hint, secrets)})
            self.session.record_hint(hint_level)

    def _log(self, slices: dict[str, Any], decision: Any, trigger: Trigger, hint_level: int | None) -> None:
        try:
            log_deliberation(
                self.log_path,
                session_id=self.session_id,
                task_id=self.session.active_id or "unknown",
                slices=slices,
                decision=decision,
                metadata={"trigger": trigger.value, "hint_level": hint_level},
            )
        except OSError:
            log.exception("[%s] could not write deliberation log", self.session_id)


async def _none() -> None:
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(
    jev: JevClient | None = None, log_path: Path = DEFAULT_LOG_PATH, slim: Publisher | None = None
) -> FastAPI:
    """Build the app. Pass `jev` (e.g. FakeJevClient) to run without the TypeSafe API.

    In the real app, SLIM publishing starts when SLIM_ENDPOINT is set (see agntcy_slim.py);
    tests can pass a fake `slim` publisher instead.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if jev is not None:
            app.state.jev, app.state.slim = jev, slim
            yield
            return
        _load_env_file(str(ROOT / ".env.local"))
        publisher = await start_publisher_from_env()
        app.state.slim = publisher
        try:
            async with TypeSafeJevClient() as client:
                app.state.jev = client
                yield
        finally:
            if publisher is not None:
                await publisher.aclose()

    app = FastAPI(title="Sherpa", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await Connection(ws, app.state.jev, log_path, app.state.slim).run()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)
