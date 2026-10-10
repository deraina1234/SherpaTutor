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
from contracts import HelpLevel, JevClient
from deliberation import InProcBus, run_deliberation
from hints import explain_deliberation_async, write_hint_async
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


def _hint_text(result: Any) -> str | None:
    """write_hint_async returns str | None; accept an object with .text too."""
    if result is None or isinstance(result, str):
        return result
    return getattr(result, "text", None)


# ---------------------------------------------------------------------------
# One browser connection
# ---------------------------------------------------------------------------


class Connection:
    def __init__(self, ws: WebSocket, jev: JevClient, log_path: Path) -> None:
        self.ws = ws
        self.jev = jev
        self.log_path = log_path
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

        bus = InProcBus()
        subscription = bus.subscribe("*")

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
            write_hint_async(context["code"], context["last_error"], level, task_for(problem_id).reference_solution_summary)
            if level > 0
            else _none(),
        )
        hint = _hint_text(hint_result)

        await self.send(
            {
                "type": "decision",
                "outcome": decision.outcome.value,
                "level": level,  # 0 for deadlock: the UI shows "No automatic hint" and "Want a nudge?"
                "explanation": redact(explanation, secrets),
            }
        )
        if hint:
            await self.send({"type": "hint", "level": level, "text": redact(hint, secrets)})
            self.session.record_hint(level)
        self._log(slices, decision, trigger, hint_level=level if hint else None)

    async def _nudge(self, problem_id: str) -> None:
        builder = self.session.active
        hints = getattr(builder, "hints", [])
        last = hints[-1].level if hints else None
        task = task_for(problem_id)
        cap = self._caps.get(problem_id, int(task.policy_max_level))
        level = nudge_level(last, cap)
        log.info("[%s] ask_nudge on %s: last hint L%s, cap L%d -> L%d", self.session_id, problem_id, last, cap, level)
        if level <= 0:
            return

        context = self.session.hint_context()
        hint = _hint_text(
            await write_hint_async(context["code"], context["last_error"], level, task.reference_solution_summary)
        )
        if hint:
            secrets = secret_values(self.session.slices())
            await self.send({"type": "hint", "level": level, "text": redact(hint, secrets)})
            self.session.record_hint(level)

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


def create_app(jev: JevClient | None = None, log_path: Path = DEFAULT_LOG_PATH) -> FastAPI:
    """Build the app. Pass `jev` (e.g. FakeJevClient) to run without the TypeSafe API."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if jev is not None:
            app.state.jev = jev
            yield
            return
        _load_env_file(str(ROOT / ".env.local"))
        async with TypeSafeJevClient() as client:
            app.state.jev = client
            yield

    app = FastAPI(title="Sherpa", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await Connection(ws, app.state.jev, log_path).run()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=8000)
