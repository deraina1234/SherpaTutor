"""Scrub secret slice values from anything that leaves the system (logs, bus).

Jev never generates text, so a secret can only leak as a verbatim copy made by
code. Exact-string replacement is therefore enough to catch it, under any key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from contracts import SECRET_SLICE_FIELDS, Bus, SliceDict

REDACTED = "[redacted]"


def secret_values(slices: Mapping[str, SliceDict]) -> list[str]:
    """Collect the non-empty string values of SECRET_SLICE_FIELDS across all slices."""
    values: list[str] = []
    for slice in slices.values():
        for field_name in SECRET_SLICE_FIELDS:
            value = slice.get(field_name)  # type: ignore[misc]
            if isinstance(value, str) and value.strip():
                values.append(value)
    # Longest first so a secret containing another is replaced whole.
    return sorted(set(values), key=len, reverse=True)


def strip_secret_fields(slice: SliceDict) -> dict[str, Any]:
    """Return a copy of a slice without SECRET_SLICE_FIELDS."""
    return {key: value for key, value in slice.items() if key not in SECRET_SLICE_FIELDS}


def redact(value: Any, secrets: list[str]) -> Any:
    """Return a copy of value with every occurrence of each secret replaced, recursively."""
    if not secrets:
        return value
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, Mapping):
        return {redact(key, secrets): redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(item, secrets) for item in value)
    return value


class RedactingBus:
    """Bus wrapper that redacts secrets from every published message."""

    def __init__(self, bus: Bus, secrets: list[str]) -> None:
        self._bus = bus
        self._secrets = secrets

    async def publish(self, topic: str, msg: dict) -> None:
        await self._bus.publish(topic, redact(msg, self._secrets))

    def subscribe(self, topic: str) -> AsyncIterator[dict]:
        return self._bus.subscribe(topic)
