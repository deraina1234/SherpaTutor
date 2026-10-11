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
    exp = explain_deliberation(decision, {"learning": {"last_run": {"passed": False, "error": "Case 2: expected 4, got 5"}}}, api_key="")
    assert exp.startswith("Your latest run failed: Case 2: expected 4, got 5.")
    assert "The agents noticed that you seem to be missing a concept" in exp
    assert "voted the same level" in exp
    assert exp.endswith("So Sherpa is giving you a conceptual hint.")


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
    assert "the same error keeps coming back" in exp
    assert "less intrusive level" in exp
    assert "Integrity's ceiling lowered a vote" in exp
    assert exp.endswith("So Sherpa is giving you a nudge.")


def test_offline_explanation_soft_disagreement_level_0() -> None:
    # Level 0 must state that no hint was given and the learner can ask for one
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
    exp = explain_deliberation(decision, {"learning": {"last_run": {"passed": True, "error": None}}}, api_key="")
    assert exp.startswith("Your tests all pass.")
    assert exp.endswith("So Sherpa is holding back for now; you can ask for help any time.")
    assert "nudge" not in exp


def test_offline_explanation_deadlock() -> None:
    decision = Decision(
        outcome=OutcomeKind.DEADLOCK,
        final_level=None,
        rounds=[[make_pos("learning", 0, 0), make_pos("momentum", 0, 3)]],
        cap_applied=False,
        ask_learner=True,
        rounds_used=1,
    )
    exp = explain_deliberation(decision, {"learning": {"last_run": {"passed": None, "error": None}}}, api_key="")
    assert exp.startswith("You haven't run your code yet.")
    assert "two or more levels apart" in exp
    assert exp.endswith("ask for a nudge whenever you want one.")


def test_explanation_closing_states_the_level_actually_given() -> None:
    from hints import frame_explanation

    decision = Decision(OutcomeKind.CONSENSUS, 2, [[make_pos("learning", 0, 2), make_pos("momentum", 0, 2)]], False, False, 1)
    downgraded = frame_explanation("Both agents agreed.", decision, None, given_level=1)
    assert downgraded == (
        "Both agents agreed. So Sherpa is giving you a nudge, a step lighter than the agents chose, "
        "because the stronger hint did not pass Sherpa's checks."
    )


def test_jev_replaces_an_explanation_that_contradicts_the_facts() -> None:
    from hints import explain_why_async

    decision = Decision(OutcomeKind.CONSENSUS, 1, [[make_pos("learning", 0, 1, tags=["concept_gap"]), make_pos("momentum", 0, 1)]], False, False, 1)
    slices = {"learning": {"last_run": {"passed": False, "error": "Case 4 failed"}}}

    async def run(probability: float) -> tuple[str, FakeJevClient]:
        jev = FakeJevClient(noul_probability=probability)
        with patch("hints.complete_text", AsyncMock(return_value="Since all your tests passed, no hint was needed.")):
            return await explain_why_async(decision, slices, jev=jev), jev

    rejected, jev = asyncio.run(run(0.9))
    assert rejected.startswith("The agents noticed that you seem to be missing a concept")
    state = jev.calls[0][1]
    assert state["run_result"] == "Your latest run failed: Case 4 failed." and state["result"] == "So Sherpa is giving you a nudge."
    kept, _ = asyncio.run(run(0.1))
    assert kept == "Since all your tests passed, no hint was needed."


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
    questions = sorted(question for _, _, question in jev.calls)
    assert len(questions) == 2  # one leak check, one grounding check
    assert "Does `hint` claim something" in questions[0] and "Does `hint` reveal the approach" in questions[1]


class ScriptedLeakJev(FakeJevClient):
    """Leak and grounding checks each return their scripted probabilities in order (default 0.1)."""

    def __init__(self, leak: list[float], ungrounded: list[float] | None = None) -> None:
        super().__init__()
        self.leak, self.ungrounded = list(leak), list(ungrounded or [])

    async def noul(self, state: dict, question: str) -> JevNoul:
        self.calls.append(("noul", state, question))
        script = self.leak if "reference_solution_summary" in state else self.ungrounded
        return JevNoul(probability=script.pop(0) if script else 0.1, confidence=0.9)


def leak_checks(jev: FakeJevClient) -> int:
    return sum(1 for _, state, _ in jev.calls if "reference_solution_summary" in state)


@pytest.mark.parametrize(
    ("probabilities", "level", "downgraded"),
    [([0.9, 0.2], HelpLevel.CONCEPTUAL_HINT, False), ([0.9, 0.9, 0.2], HelpLevel.NUDGE, True)],
)
def test_jev_verification_rewrites_leaked_hint_then_downgrades(
    probabilities: list[float], level: int, downgraded: bool
) -> None:
    jev = ScriptedLeakJev(probabilities)
    with patch("hints._generate_raw_hint", AsyncMock(side_effect=lambda **kw: (f"draft {leak_checks(jev)}", True))) as gen:
        hint = write_hint(
            code="nums = [1, 2]",
            error="IndexError",
            help_level=HelpLevel.CONCEPTUAL_HINT,  # level 2
            reference_solution_summary="binary search",
            api_key="",
            jev=jev,
        )
    assert (hint.verified, hint.level, hint.downgraded) == (True, level, downgraded)
    assert leak_checks(jev) == len(probabilities)
    # Each rewrite is shown the draft rejected just before it, and why.
    rewrites = gen.call_args_list[1:]
    assert [call.kwargs["rejected_draft"] for call in rewrites] == [f"draft {i}" for i in range(len(rewrites))]
    assert all(call.kwargs["rejected_reason"] == "leak" for call in rewrites)


def test_jev_verification_rewrites_an_ungrounded_hint() -> None:
    jev = ScriptedLeakJev(leak=[], ungrounded=[0.9])  # first draft invents a bug; the rewrite is fine
    drafts = iter(["Your indentation is wrong on line 2.", "Your function returns 3 for [1, 2]."])
    with patch("hints._generate_raw_hint", AsyncMock(side_effect=lambda **kw: (next(drafts), True))) as gen:
        hint = write_hint("def f(): pass", "Case 1: expected 2, got 3", HelpLevel.NUDGE, "summary", api_key="", jev=jev)
    assert (hint.text, hint.level, hint.verified, hint.downgraded) == ("Your function returns 3 for [1, 2].", 1, True, False)
    assert gen.call_args_list[1].kwargs["rejected_reason"] == "ungrounded"
    grounding = [state for _, state, _ in jev.calls if "run_result" in state]
    assert grounding[0]["code"] == "def f(): pass" and "expected 2, got 3" in grounding[0]["run_result"]


def test_ungrounded_rejection_reason_goes_into_the_prompt() -> None:
    from hints import _build_hint_prompt

    prompt = _build_hint_prompt("code", "err", HelpLevel.NUDGE, "summary", False, "bad draft", None, "ungrounded")
    assert "bad draft" in prompt and "do not show" in prompt and "reveals less" not in prompt


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


def test_hint_prompt_includes_the_problem_and_asks_about_the_approach() -> None:
    from hints import _build_hint_prompt

    prompt = _build_hint_prompt(
        "def three_sum(nums):\n    pass\n", None, HelpLevel.NUDGE, "secret", None, problem_statement="3Sum: Return every triplet."
    )
    assert "PROBLEM:\n3Sum: Return every triplet." in prompt
    assert "APPROACH" in prompt and "still typing" in prompt
    assert "PROBLEM:" not in _build_hint_prompt("x = 1", None, HelpLevel.NUDGE)
