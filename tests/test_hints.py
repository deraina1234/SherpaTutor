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


class ScriptedLeakJev(FakeJevClient):
    """Leak checks return the given probabilities in order."""

    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.probabilities = probabilities

    async def noul(self, state: dict, question: str) -> JevNoul:
        self.calls.append(("noul", state, question))
        return JevNoul(probability=self.probabilities[len(self.calls) - 1], confidence=0.9)


@pytest.mark.parametrize(
    ("probabilities", "level", "downgraded"),
    [([0.9, 0.2], HelpLevel.CONCEPTUAL_HINT, False), ([0.9, 0.9, 0.2], HelpLevel.NUDGE, True)],
)
def test_jev_verification_rewrites_leaked_hint_then_downgrades(
    probabilities: list[float], level: int, downgraded: bool
) -> None:
    jev = ScriptedLeakJev(probabilities)
    with patch("hints._generate_raw_hint", AsyncMock(side_effect=lambda **kw: (f"draft {len(jev.calls)}", True))) as gen:
        hint = write_hint(
            code="nums = [1, 2]",
            error="IndexError",
            help_level=HelpLevel.CONCEPTUAL_HINT,  # level 2
            reference_solution_summary="binary search",
            api_key="",
            jev=jev,
        )
    assert (hint.verified, hint.level, hint.downgraded) == (True, level, downgraded)
    assert len(jev.calls) == len(probabilities)
    # Every rewrite is shown the first leaked draft so it can avoid it.
    assert all(call.kwargs["rejected_draft"] == "draft 0" for call in gen.call_args_list[1:])


def test_rejected_draft_goes_into_the_prompt() -> None:
    from hints import _build_hint_prompt

    prompt = _build_hint_prompt("code", "err", HelpLevel.NUDGE, "summary", False, rejected_draft="check the stack at the end")
    assert "check the stack at the end" in prompt and "reveals less" in prompt


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


def test_hint_prompt_tells_the_writer_when_tests_pass() -> None:
    from hints import _build_hint_prompt

    passing = _build_hint_prompt("code", None, HelpLevel.NUDGE, last_passed=True)
    assert "All test cases pass" in passing and "do not claim it has a bug" in passing
    assert "not run their code yet" in _build_hint_prompt("code", None, HelpLevel.NUDGE)
    assert "Case 2: expected 4" in _build_hint_prompt("code", "Case 2: expected 4", HelpLevel.NUDGE, last_passed=False)
    assert "Never invent a bug" in passing


def test_offline_hint_for_passing_code_does_not_invent_a_bug() -> None:
    from hints import ALL_TESTS_PASS_TEXT, _offline_hint

    assert _offline_hint("code", None, HelpLevel.NUDGE, last_passed=True) == ALL_TESTS_PASS_TEXT


def test_explainer_prompt_lists_only_real_facts() -> None:
    from contracts import Argument
    from hints import _build_explainer_prompt

    idle = Argument(id="momentum.r0.long_idle", agent="momentum", round=0, claim="long_idle", direction=1)
    rebuttal = Argument(
        id="learning.r1.re.momentum.r0.long_idle", agent="learning", round=1, claim="long_idle", direction=0,
        evidence={"responds_to": "momentum.r0.long_idle", "holds": 0.2}, attacks="momentum.r0.long_idle",
    )
    round0 = [
        make_pos("learning", 0, 0),
        Position(agent="momentum", round=0, level=2, confidence=0.8, reason_tags=["long_idle"], arguments=[idle]),
        make_pos("integrity", 0, 2, cap=2),
    ]
    round1 = [
        Position(agent="learning", round=1, level=0, confidence=0.85, arguments=[rebuttal]),
        round0[1],
        round0[2],
    ]
    decision = Decision(OutcomeKind.DEADLOCK, None, [round0, round1], cap_applied=False, ask_learner=True, rounds_used=2)
    prompt = _build_explainer_prompt(decision, {"learning": {"last_run": {"passed": True, "error": None}}})

    assert "Momentum first voted for a conceptual hint, judging from your code and test results." in prompt
    assert "Signals it noticed: you had stopped editing for a while." in prompt
    assert "Learning rejected Momentum's point that you had stopped editing for a while." in prompt
    assert "Integrity set a ceiling of a conceptual hint (it does not vote)." in prompt
    assert "the ceiling changed nothing" in prompt
    assert "passed every test" in prompt


def test_explainer_prompt_quotes_the_agents_written_arguments() -> None:
    from contracts import Argument
    from hints import _build_explainer_prompt

    idle = Argument(
        id="momentum.r0.long_idle", agent="momentum", round=0, claim="long_idle", direction=1,
        text="No edits for 240 seconds.",
    )
    rebuttal = Argument(
        id="learning.r1.re.momentum.r0.long_idle", agent="learning", round=1, claim="long_idle", direction=0,
        evidence={"responds_to": "momentum.r0.long_idle", "holds": 0.2}, attacks="momentum.r0.long_idle",
        text="They ran the tests 40 seconds ago.",
    )
    momentum = Position(agent="momentum", round=0, level=2, confidence=0.8, reason_tags=["long_idle"], arguments=[idle])
    round0 = [make_pos("learning", 0, 0), momentum, make_pos("integrity", 0, 2, cap=2)]
    round1 = [Position(agent="learning", round=1, level=0, confidence=0.85, arguments=[rebuttal]), momentum, round0[2]]
    decision = Decision(OutcomeKind.DEADLOCK, None, [round0, round1], cap_applied=False, ask_learner=True, rounds_used=2)
    prompt = _build_explainer_prompt(decision)

    assert 'Momentum argued: "No edits for 240 seconds."' in prompt
    assert 'rejected Momentum\'s point that you had stopped editing for a while. It said: "They ran the tests 40 seconds ago."' in prompt
