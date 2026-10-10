"""Tests for hints.py (LLM Hint Writer & Deliberation Explainer)."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contracts import Decision, HelpLevel, JevNoul, OutcomeKind, Position
from hints import (
    HintResult,
    explain_deliberation,
    explain_deliberation_async,
    write_hint,
    write_hint_async,
)
from jev import FakeJevClient


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
    hint = write_hint("def f(): pass", "IndexError", HelpLevel.NONE, api_key="")
    assert hint.text is None
    assert hint.level == 0
    assert not hint.verified
    assert not hint.downgraded
    assert not bool(hint)


@pytest.mark.parametrize("level", [HelpLevel.NUDGE, HelpLevel.CONCEPTUAL_HINT, HelpLevel.POINT_TO_AREA])
def test_offline_hint_generation(level: int) -> None:
    hint = write_hint("def f(): pass", "IndexError: list index out of range", level, api_key="")
    assert hint.text is not None
    assert len(hint.text) > 10
    assert "IndexError" in hint.text
    assert hint.level == level
    assert not hint.verified
    assert not hint.downgraded


def test_offline_explanation_consensus() -> None:
    decision = Decision(
        outcome=OutcomeKind.CONSENSUS,
        final_level=2,
        rounds=[[make_pos("learning", 0, 2, tags=["concept_gap"]), make_pos("momentum", 0, 2)]],
        cap_applied=False,
        ask_learner=False,
        rounds_used=1,
    )
    exp = explain_deliberation(decision, api_key="")
    assert "consensus" in exp
    assert "Level 2" in exp
    assert "concept" in exp


def test_offline_explanation_soft_disagreement_level_1() -> None:
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
    exp = explain_deliberation(decision, api_key="")
    assert "soft disagreement" in exp
    assert "least intrusive" in exp
    assert "Level 1" in exp
    assert "Integrity clamped" in exp


def test_offline_explanation_soft_disagreement_level_0() -> None:
    # Level 0 soft disagreement must state that no hint was given and the learner can ask for one
    decision = Decision(
        outcome=OutcomeKind.SOFT_DISAGREEMENT,
        final_level=0,
        rounds=[
            [make_pos("learning", 0, 0), make_pos("momentum", 0, 1)],
            [make_pos("learning", 1, 0), make_pos("momentum", 1, 0)],
        ],
        cap_applied=False,
        ask_learner=False,
        rounds_used=2,
    )
    exp = explain_deliberation(decision, api_key="")
    assert "soft disagreement" in exp
    assert "no hint was given" in exp
    assert "ask for one" in exp
    assert "light nudge" not in exp


def test_offline_explanation_deadlock() -> None:
    decision = Decision(
        outcome=OutcomeKind.DEADLOCK,
        final_level=None,
        rounds=[[make_pos("learning", 0, 0), make_pos("momentum", 0, 3)]],
        cap_applied=False,
        ask_learner=True,
        rounds_used=1,
    )
    exp = explain_deliberation(decision, api_key="")
    assert "deadlock" in exp
    assert "stayed silent" in exp


def test_fallback_logs_warning_on_missing_key(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        # Explicitly pass api_key="" to trigger missing key fallback
        res = write_hint("code", "error", HelpLevel.NUDGE, api_key="")
        assert res.text is not None

    assert any("Falling back to template hint: GEMINI_API_KEY is not set." in record.message for record in caplog.records)


def test_fallback_logs_warning_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        mock_models = AsyncMock()
        mock_models.generate_content.side_effect = RuntimeError("Network timeout simulation")

        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        with patch("google.genai.Client", return_value=mock_client):
            with caplog.at_level(logging.WARNING):
                res = await write_hint_async("code", "error", HelpLevel.NUDGE, api_key="dummy-key")
                assert res.text is not None

        assert any("Falling back to template hint due to RuntimeError: Network timeout simulation" in record.message for record in caplog.records)

    asyncio.run(run())


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
            assert hint.text == "Look at line 3 where the array bounds are exceeded."
            assert hint.level == HelpLevel.POINT_TO_AREA
            assert not hint.verified
            mock_models.generate_content.assert_called_once()

    asyncio.run(run())


def test_jev_verification_passes_with_safe_hint() -> None:
    # Jev checks the generated hint with probability 0.1 (safe)
    jev = FakeJevClient(noul_probability=0.1)
    hint = write_hint(
        code="nums = [1, 2]",
        error="IndexError",
        help_level=HelpLevel.CONCEPTUAL_HINT,
        reference_solution_summary="binary search",
        api_key="",
        jev=jev,
    )
    assert hint.verified is True
    assert hint.downgraded is False
    assert hint.level == HelpLevel.CONCEPTUAL_HINT
    assert len(jev.calls) == 1
    assert jev.calls[0][0] == "noul"
    assert "Does `hint` reveal the approach" in jev.calls[0][2]


def test_jev_verification_downgrades_leaked_hint() -> None:
    # Custom client where first call returns 0.9 (leak), second call returns 0.2 (safe)
    class DynamicJevClient(FakeJevClient):
        def __init__(self) -> None:
            super().__init__()
            self.noul_calls = 0

        async def noul(self, state: dict, question: str) -> JevNoul:
            self.calls.append(("noul", state, question))
            self.noul_calls += 1
            prob = 0.9 if self.noul_calls == 1 else 0.2
            return JevNoul(probability=prob, confidence=0.9)

    jev = DynamicJevClient()
    hint = write_hint(
        code="nums = [1, 2]",
        error="IndexError",
        help_level=HelpLevel.CONCEPTUAL_HINT,  # level 2
        reference_solution_summary="binary search",
        api_key="",
        jev=jev,
    )
    # Leaked at level 2 -> downgraded to level 1 (NUDGE) and verified safe!
    assert hint.verified is True
    assert hint.downgraded is True
    assert hint.level == HelpLevel.NUDGE
    assert jev.noul_calls == 2


def test_jev_verification_fails_downgrade_returns_template() -> None:
    # Jev consistently flags hints as leaks (prob 0.9)
    jev = FakeJevClient(noul_probability=0.9)
    hint = write_hint(
        code="nums = [1, 2]",
        error="IndexError",
        help_level=HelpLevel.CONCEPTUAL_HINT,
        reference_solution_summary="binary search",
        api_key="",
        jev=jev,
    )
    # Both level 2 and level 1 leak -> returns template hint for original level
    assert hint.verified is False
    assert hint.downgraded is True
    assert hint.level == HelpLevel.CONCEPTUAL_HINT
    assert hint.text is not None
