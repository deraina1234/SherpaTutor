"""JSONL logger for Sherpa deliberations.

Logs session snapshots, private slices, and full Decision objects with rounds
for replayable evaluations, baseline comparisons, and debugging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts import Decision, SliceDict


def log_deliberation(
    filepath: str | Path,
    session_id: str,
    task_id: str,
    slices: dict[str, SliceDict],
    decision: Decision,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a deliberation record to a JSONL file.

    Returns the logged record dict.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "task_id": task_id,
        "slices": {agent_name: dict(sl) for agent_name, sl in slices.items()},
        "decision": decision.to_dict(),
        "metadata": metadata or {},
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def read_deliberations(filepath: str | Path) -> list[dict[str, Any]]:
    """Read all logged deliberation records from a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
