"""AGNTCY SLIM transport for Sherpa's deliberation messages.

Every round and decision the deliberation loop publishes is also sent to an
end-to-end encrypted (MLS) SLIM group channel, so other processes, such as an
instructor dashboard or an audit logger, can watch deliberations live. Messages
pass through the loop's redaction first, so reference solutions never reach SLIM.

    python agntcy_slim.py node        # a local SLIM node (no Docker needed)
    python agntcy_slim.py observer    # joins the channel and prints/logs messages
    python server.py                  # with SLIM_ENDPOINT set, publishes to the channel

Configuration (environment or .env.local):
    SLIM_ENDPOINT       host:port of the SLIM node, e.g. 127.0.0.1:46357 (unset = SLIM off)
    SLIM_SHARED_SECRET  shared secret for app identity (dev only; use JWT/SPIRE in production)
    SLIM_CHANNEL        group channel name, default agntcy/sherpa/deliberations
    SLIM_OBSERVERS      comma-separated participant names to invite, default agntcy/sherpa/observer
    SLIM_APP_NAME       this server's name, default agntcy/sherpa/server
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from deliberation import InProcBus

log = logging.getLogger("sherpa.slim")

DEFAULT_CHANNEL = "agntcy/sherpa/deliberations"
DEFAULT_OBSERVER = "agntcy/sherpa/observer"
DEFAULT_APP_NAME = "agntcy/sherpa/server"
INVITE_TIMEOUT_SECONDS = 5.0
REINVITE_EVERY_SECONDS = 10.0
MIN_SECRET_LENGTH = 32
OBSERVER_LOG = Path(__file__).resolve().parent / "logs" / "slim_observer.jsonl"


@dataclass(frozen=True)
class SlimSettings:
    endpoint: str
    shared_secret: str
    channel: str = DEFAULT_CHANNEL
    observers: tuple[str, ...] = (DEFAULT_OBSERVER,)
    app_name: str = DEFAULT_APP_NAME

    @classmethod
    def from_env(cls) -> SlimSettings | None:
        endpoint = os.getenv("SLIM_ENDPOINT", "").strip()
        if not endpoint:
            return None
        secret = os.getenv("SLIM_SHARED_SECRET", "").strip()
        if len(secret) < MIN_SECRET_LENGTH:
            # SLIM rejects short keys with "hmac key is too short"; say so up front.
            raise ValueError(
                f"SLIM_SHARED_SECRET must be at least {MIN_SECRET_LENGTH} characters "
                '(try: python -c "import secrets; print(secrets.token_urlsafe(36))")'
            )
        observers = tuple(
            name.strip() for name in os.getenv("SLIM_OBSERVERS", DEFAULT_OBSERVER).split(",") if name.strip()
        )
        return cls(
            endpoint=endpoint,
            shared_secret=secret,
            channel=os.getenv("SLIM_CHANNEL", DEFAULT_CHANNEL).strip() or DEFAULT_CHANNEL,
            observers=observers,
            app_name=os.getenv("SLIM_APP_NAME", DEFAULT_APP_NAME).strip() or DEFAULT_APP_NAME,
        )


# ---------------------------------------------------------------------------
# SLIM plumbing (slim_bindings is imported lazily: SLIM is optional)
# ---------------------------------------------------------------------------


def _service() -> Any:
    import slim_bindings

    if not slim_bindings.is_initialized():
        slim_bindings.initialize_with_configs(
            tracing_config=slim_bindings.new_tracing_config(),
            runtime_config=slim_bindings.new_runtime_config(),
            service_config=[slim_bindings.new_service_config()],
        )
    return slim_bindings.get_global_service()


async def _connect_app(settings: SlimSettings, name: str) -> tuple[Any, int]:
    import slim_bindings

    service = _service()
    conn_id = await service.connect_async(slim_bindings.new_insecure_client_config(f"http://{settings.endpoint}"))
    local = slim_bindings.Name.from_string(name)
    app = service.create_app_with_secret(local, settings.shared_secret)
    await app.subscribe_async(local, conn_id)
    return app, conn_id


class Publisher(Protocol):
    async def publish(self, envelope: dict[str, Any]) -> None: ...


@dataclass
class SlimPublisher:
    """Owns this server's SLIM app and an MLS-encrypted group session on the channel."""

    settings: SlimSettings
    _app: Any = None
    _conn_id: int = 0
    _session: Any = None
    _pending: set[str] = field(default_factory=set)
    _last_invite_attempt: float = 0.0

    async def start(self) -> None:
        import slim_bindings

        self._app, self._conn_id = await _connect_app(self.settings, self.settings.app_name)
        config = slim_bindings.SessionConfig(
            session_type=slim_bindings.SessionType.GROUP,
            max_retries=5,
            interval=datetime.timedelta(seconds=1),
            metadata={},
            mls_settings=slim_bindings.MlsSettings(
                header_integrity_validation_percent=100, max_seen_control_message_ids_size=None
            ),
        )
        created = self._app.create_session(config, slim_bindings.Name.from_string(self.settings.channel))
        await created.completion.wait_async()
        self._session = created.session
        self._pending = set(self.settings.observers)
        await self._invite_pending()
        log.info("SLIM group %s ready on %s", self.settings.channel, self.settings.endpoint)

    async def _invite_pending(self) -> None:
        """Invite observers not yet in the group; ones that aren't running are retried later."""
        import slim_bindings

        self._last_invite_attempt = time.monotonic()
        for name in sorted(self._pending):
            peer = slim_bindings.Name.from_string(name)
            try:
                await self._app.set_route_async(peer, self._conn_id)
                handle = await self._session.invite_async(peer)
                await asyncio.wait_for(handle.wait_async(), INVITE_TIMEOUT_SECONDS)
            except Exception as err:  # noqa: BLE001 - observers are optional; never break deliberation
                if "already in group" not in str(err):
                    log.info("SLIM observer %s not reachable yet (%s); will retry", name, type(err).__name__)
                    continue
            self._pending.discard(name)
            log.info("SLIM observer %s joined %s", name, self.settings.channel)

    async def publish(self, envelope: dict[str, Any]) -> None:
        if self._session is None:
            return
        if self._pending and time.monotonic() - self._last_invite_attempt > REINVITE_EVERY_SECONDS:
            await self._invite_pending()
        try:
            await self._session.publish_async(json.dumps(envelope).encode(), None, None)
        except Exception as err:  # noqa: BLE001 - SLIM is a side channel; the learner's UI must not depend on it
            log.warning("SLIM publish failed (%s: %s)", type(err).__name__, err)

    async def aclose(self) -> None:
        if self._app is not None and self._session is not None:
            try:
                handle = await self._app.delete_session_async(self._session)
                await asyncio.wait_for(handle.wait_async(), INVITE_TIMEOUT_SECONDS)
            except Exception:  # noqa: BLE001
                pass
        self._session = None


async def start_publisher_from_env() -> SlimPublisher | None:
    """A started publisher when SLIM_ENDPOINT is set, else None. Failures are logged, not raised."""
    try:
        settings = SlimSettings.from_env()
        if settings is None:
            return None
        publisher = SlimPublisher(settings)
        await publisher.start()
        return publisher
    except Exception as err:  # noqa: BLE001
        log.warning("SLIM disabled: could not start publisher (%s: %s)", type(err).__name__, err)
        return None


# ---------------------------------------------------------------------------
# Bus that delivers locally and over SLIM
# ---------------------------------------------------------------------------


class FanoutBus:
    """contracts.Bus: delivers to local subscribers and also publishes to a SLIM channel.

    `context` (session, problem, trigger) is attached to every SLIM message, since one
    channel carries deliberations from every learner session.
    """

    def __init__(self, local: InProcBus, remote: Publisher, context: dict[str, Any]) -> None:
        self._local = local
        self._remote = remote
        self._context = context

    async def publish(self, topic: str, msg: dict) -> None:
        await self._local.publish(topic, msg)
        await self._remote.publish({"topic": topic, "context": self._context, "message": msg})

    def subscribe(self, topic: str) -> AsyncIterator[dict]:
        return self._local.subscribe(topic)


# ---------------------------------------------------------------------------
# CLI: local node and observer
# ---------------------------------------------------------------------------


async def run_node(endpoint: str) -> None:
    import slim_bindings

    service = _service()
    await service.run_server_async(slim_bindings.new_insecure_server_config(endpoint))
    print(f"SLIM node listening on {endpoint} (Ctrl+C to stop)", flush=True)
    await asyncio.Event().wait()


def describe(envelope: dict[str, Any]) -> str:
    ctx, msg = envelope.get("context", {}), envelope.get("message", {})
    where = f"[{ctx.get('session_id', '?')} {ctx.get('problem_id', '?')} {ctx.get('trigger', '?')}]"
    if "positions" in msg:
        votes = "  ".join(
            f"{p['agent']}=L{p['level']}" + (f"(cap {p['cap']})" if p.get("cap") is not None else "")
            for p in msg["positions"]
        )
        return f"{where} round {msg.get('round')}: {votes}"
    if "outcome" in msg:
        return f"{where} decision: {msg['outcome']} at L{msg.get('final_level')} after {msg.get('rounds_used')} round(s)"
    return f"{where} {envelope.get('topic')}"


async def run_observer(settings: SlimSettings, name: str) -> None:
    app, _ = await _connect_app(settings, name)
    print(f"{name} waiting for an invite to {settings.channel} ...", flush=True)
    OBSERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        session = await app.listen_for_session_async(None)
        print(f"joined {session.destination()}", flush=True)
        while True:
            try:
                received = await session.get_message_async(timeout=datetime.timedelta(seconds=60))
            except Exception as err:  # noqa: BLE001
                if "closed" in str(err).lower():
                    print("session closed; waiting for the next invite", flush=True)
                    break
                continue  # timeout: keep listening
            envelope = json.loads(received.payload.decode())
            print(describe(envelope), flush=True)
            with OBSERVER_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(envelope) + "\n")


def main() -> None:
    from jev import _load_env_file

    _load_env_file(str(Path(__file__).resolve().parent / ".env.local"))
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "node":
        asyncio.run(run_node(os.getenv("SLIM_ENDPOINT", "127.0.0.1:46357")))
    elif command == "observer":
        settings = SlimSettings.from_env()
        if settings is None:
            sys.exit("Set SLIM_ENDPOINT and SLIM_SHARED_SECRET first.")
        name = sys.argv[2] if len(sys.argv) > 2 else settings.observers[0]
        asyncio.run(run_observer(settings, name))
    else:
        sys.exit("usage: python agntcy_slim.py node | observer [name]")


if __name__ == "__main__":
    main()
