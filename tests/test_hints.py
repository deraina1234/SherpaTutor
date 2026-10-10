"""Tests for hints.py (LLM Hint Writer & Deliberation Explainer)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contracts import Decision, HelpLevel, OutcomeKind, Position
from hints import (
    explain_deliberation,
    explain_deliberation_async,
    write_hint,
    write_hint_async,
)


def make_pos(agent: str, round: int, level: int, cap: int | None = None, tags: list[str] | None = None) -> Position:
    return Position(
        agent=agent,
        round=round,
        level=level,
        confidence=0.85,
        reason_tags=tags or [],
        cap=cap,
    )


def test_write_hint_level_0_returns_none() -> None:
    hint = write_hint("def f(): pass", "IndexError", HelpLevel.NONE)
    assert hint is None


@pytest.mark.parametrize("level", [HelpLevel.NUDGE, HelpLevel.CONCEPTUAL_HINT, HelpLevel.POINT_TO_AREA])
def test_offline_hint_generation(level: int) -> None:
    hint = write_hint("def f(): pass", "IndexError: list index out of range", level)
    assert hint is not None
    assert len(hint) > 10
    assert "IndexError" in hint


def test_offline_explanation_consensus() -> None:
    decision = Decision(
        outcome=OutcomeKind.CONSENSUS,
        final_level=2,
        rounds=[[make_pos("learning", 0, 2, tags=["concept_gap"]), make_pos("momentum", 0, 2)]],
        cap_applied=False,
        ask_learner=False,
        rounds_used=1,
    )
    exp = explain_deliberation(decision)
    assert "consensus" in exp
    assert "Level 2" in exp
    assert "concept" in exp


def test_offline_explanation_soft_disagreement() -> None:
    decision = Decision(
        outcome=OutcomeKind.SOFT_DISAGREEMENT,
        final_level=1,
        rounds=[
            [make_pos("learning", 0, 1), make_pos("momentum", 0, 3, tags=["repeated_error"])],
            [make_pos("learning", 1, 1), make_pos("momentum", 1, 2)],
        ],
        cap_applied=True,
        ask_learner=False,
        rounds_used=2,
    )
    exp = explain_deliberation(decision)
    assert "soft disagreement" in exp
    assert "least intrusive" in exp
    assert "Level 1" in exp
    assert "Integrity clamped" in exp


def test_offline_explanation_deadlock() -> None:
    decision = Decision(
        outcome=OutcomeKind.DEADLOCK,
        final_level=None,
        rounds=[[make_pos("learning", 0, 0), make_pos("momentum", 0, 3)]],
        cap_applied=False,
        ask_learner=True,
        rounds_used=1,
    )
    exp = explain_deliberation(decision)
    assert "deadlock" in exp
    assert "stayed silent" in exp


def test_gemini_api_mocked_success() -> None:
    async def run() -> None:
        mock_resp = MagicMock()
        mock_resp.text = "Look at line 3 where the array bounds are exceeded."

        mock_models = AsyncMock()
        mock_models.generate_content.return_value = mock_resp

        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        with patch("google.genai.Client", return_value=mock_client):
            hint = await write_hint_async(
                code="x = [1, 2][5]",
                error="IndexError",
                help_level=HelpLevel.POINT_TO_AREA,
                api_key="test-key",
            )
            assert hint == "Look at line 3 where the array bounds are exceeded."
            mock_models.generate_content.assert_called_once()

    asyncio.run(run())
