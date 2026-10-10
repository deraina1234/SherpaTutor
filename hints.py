"""LLM Hint Writer & Deliberation Explainer for Sherpa.

Per Section 10 of the Spotter design doc:
- Hint writer: Receives code, error, agreed level, and reference solution policy.
  Generates a pedagogical hint strictly adhering to the agreed level (Levels 0-4).
- Leak checking: When Jev and reference solution summary are provided, Jev verifies
  the generated hint does not leak the solution. If a leak is detected (prob > 0.5),
  it attempts generating at one level lower, or falls back to template.
- Explainer: Turns structured deliberation rounds, reason tags, and decisions into
  2-3 plain-language sentences for the side panel.
- The LLM never decides whether to intervene; it only executes after the multi-agent
  deliberation finishes.

Powered by Google Gemini (google-genai SDK), with deterministic offline fallback
templates for testing and hackathon demo reliability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import Decision, HelpLevel, JevClient, OutcomeKind

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class HintResult:
    """Result of hint generation with leak-verification and downgrade metadata."""

    text: str | None
    level: int
    verified: bool = False
    downgraded: bool = False

    def __bool__(self) -> bool:
        return self.text is not None

    def __str__(self) -> str:
        return self.text or ""


def _load_env_file(path: str = ".env.local") -> None:
    """Load KEY=VALUE from env file if present."""
    p = Path(path)
    if not p.exists():
        p = Path(".env")
        if not p.exists():
            return
    try:
        with open(p, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except Exception:
        pass


_load_env_file()


def _get_api_key(explicit_key: str | None = None) -> str | None:
    if explicit_key is not None:
        return explicit_key.strip() or None
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# ---------------------------------------------------------------------------
# Offline Template Fallbacks (for offline testing & emergency demo backup)
# ---------------------------------------------------------------------------


def _offline_hint(code: str, error: str | None, help_level: int) -> str | None:
    """Generate a high-quality deterministic hint when offline."""
    if help_level == HelpLevel.NONE:
        return None

    err_str = f"the error `{error}`" if error else "unexpected output"

    if help_level == HelpLevel.NUDGE:
        return f"Take a close look at {err_str}. What condition in your logic causes this symptom?"
    elif help_level == HelpLevel.CONCEPTUAL_HINT:
        return (
            f"Think about the algorithm's edge cases that trigger {err_str}. "
            "Consider tracking state with intermediate variables or checking boundary conditions before indexing."
        )
    elif help_level == HelpLevel.POINT_TO_AREA:
        return (
            f"Examine the lines handling the core loop/conditions producing {err_str}. "
            "The values being compared or returned there differ from the expected test case."
        )
    elif help_level == HelpLevel.NEAR_SOLUTION:
        return (
            f"To resolve {err_str}, verify your return statements and boundary guards. "
            "Updating the accumulator or loop condition directly fixes the failing test."
        )
    return None


def _offline_explanation(decision: Decision, slices: dict[str, Any] | None = None) -> str:
    """Generate a concise plain-English explanation of the decision when offline."""
    outcome = decision.outcome
    final_level = decision.final_level
    rounds_count = decision.rounds_used

    # Gather reason tags from round 0 proposals
    tags: list[str] = []
    if decision.rounds and decision.rounds[0]:
        for p in decision.rounds[0]:
            tags.extend(p.reason_tags)
    unique_tags = list(dict.fromkeys(tags))

    tag_desc = ""
    if "repeated_error" in unique_tags:
        tag_desc += "repeated test failures were detected, "
    if "long_idle" in unique_tags:
        tag_desc += "the learner was idle for an extended period, "
    if "concept_gap" in unique_tags:
        tag_desc += "a conceptual misunderstanding was identified, "
    if "active_progress" in unique_tags:
        tag_desc += "the learner was making active progress, "

    if tag_desc:
        tag_desc = f"Noting that {tag_desc.rstrip(', ')}, "

    if outcome == OutcomeKind.CONSENSUS:
        level_str = (
            "to withhold help so you can continue exploring"
            if final_level == 0
            else f"to offer a Level {final_level} hint"
        )
        return (
            f"{tag_desc}the agents deliberated for {rounds_count} round(s) and reached consensus {level_str}. "
            f"Learning and Momentum agreed this level best supports your learning without interrupting your flow."
        )
    elif outcome == OutcomeKind.SOFT_DISAGREEMENT:
        cap_note = " Integrity clamped higher assistance to protect the solution." if decision.cap_applied else ""
        if final_level == 0 or final_level is None:
            return (
                f"{tag_desc}the agents had a soft disagreement between their proposals.{cap_note} "
                "Per protocol, Sherpa chose the least intrusive action: no hint was given and you can ask for one when ready."
            )
        else:
            return (
                f"{tag_desc}the agents had a soft disagreement between their proposals.{cap_note} "
                f"Per protocol, Sherpa chose the least intrusive action (Level {final_level}) to give you a light nudge first, "
                "leaving you in control if you need more help."
            )
    else:  # DEADLOCK
        return (
            f"{tag_desc}the agents reached a deadlock after {rounds_count} rounds with conflicting views on intervention. "
            "Rather than interrupting prematurely, Sherpa stayed silent. Feel free to request a nudge whenever you are ready."
        )


# ---------------------------------------------------------------------------
# Core LLM Functions
# ---------------------------------------------------------------------------


def _build_hint_prompt(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
) -> str:
    level_instructions = {
        HelpLevel.NUDGE: (
            "Goal: Write a short NUDGE (1-2 sentences).\n"
            "- Point the learner's attention to their error, syntax, or unexpected output.\n"
            "- Ask a guiding question.\n"
            "- DO NOT explain the concept or algorithm.\n"
            "- DO NOT write or suggest code."
        ),
        HelpLevel.CONCEPTUAL_HINT: (
            "Goal: Write a CONCEPTUAL HINT (1-3 sentences).\n"
            "- Explain the core programming concept, algorithm invariant, or mental model they are missing.\n"
            "- Use plain English explanation.\n"
            "- DO NOT write code snippets or give away line numbers."
        ),
        HelpLevel.POINT_TO_AREA: (
            "Goal: POINT TO THE AREA (1-3 sentences).\n"
            "- Point out the specific line, loop, or conditional where their bug occurs.\n"
            "- Explain what is happening on that line vs what was intended.\n"
            "- DO NOT provide the complete replacement code."
        ),
        HelpLevel.NEAR_SOLUTION: (
            "Goal: Write a NEAR-SOLUTION (2-3 sentences).\n"
            "- Outline the fix or provide pseudo-code structure, leaving the final step to the learner."
        ),
    }

    instructions = level_instructions.get(help_level, level_instructions[HelpLevel.NUDGE])

    reference_constraint = ""
    if reference_solution_summary:
        reference_constraint = (
            f"\nCRITICAL INTEGRITY RULE: You must NOT reveal or copy this reference solution:\n"
            f"\"{reference_solution_summary}\"\n"
            f"Never reveal the full answer directly."
        )

    return (
        "You are an encouraging, pedagogical AI programming tutor helping a student.\n"
        f"{instructions}\n"
        f"{reference_constraint}\n\n"
        f"STUDENT CODE:\n```python\n{code}\n```\n\n"
        f"ERROR / TEST FAILURE:\n{error or 'None (learner is stuck or idle)'}\n\n"
        "Respond with ONLY the hint text. Keep it concise, friendly, and pedagogically sound."
    )


def _build_explainer_prompt(decision: Decision, slices: dict[str, Any] | None = None) -> str:
    rounds_summary: list[str] = []
    for r_idx, r_positions in enumerate(decision.rounds):
        round_entries = [
            f"  - {p.agent}: Level {p.level} (conf: {p.confidence:.2f}, tags: {p.reason_tags})"
            for p in r_positions
        ]
        rounds_summary.append(f"Round {r_idx}:\n" + "\n".join(round_entries))

    rounds_text = "\n".join(rounds_summary)

    level_0_soft_note = ""
    if decision.outcome == OutcomeKind.SOFT_DISAGREEMENT and decision.final_level == 0:
        level_0_soft_note = (
            "\nNote: The final agreed level is 0, so no hint was given. "
            "Explain that no hint was given to let the student continue independently, and they can request help if needed."
        )

    return (
        "You are the explainer component of Sherpa, a multi-agent coding tutor.\n"
        "Three peer agents with competing goals just deliberated on how much help to give a student:\n"
        "- Learning: protects long-term learning and productive struggle.\n"
        "- Momentum: minimizes frustration and steps in before stalling out.\n"
        "- Integrity: ensures hints never reveal the core task solution.\n\n"
        f"Outcome: {decision.outcome.value}\n"
        f"Final Help Level: {decision.final_level}\n"
        f"Integrity Cap Applied: {decision.cap_applied}\n"
        f"Ask Learner (Deadlock): {decision.ask_learner}\n"
        f"{level_0_soft_note}\n\n"
        f"Deliberation History:\n{rounds_text}\n\n"
        "Write 2-3 plain-language, encouraging sentences explaining to the student why the agents took "
        "their stances and why this final outcome was chosen. "
        "Highlight any conflict, consensus, or cap naturally without sounding overly robotic."
    )


async def _generate_raw_hint(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None,
    api_key: str | None,
    model: str,
) -> tuple[str, bool]:
    """Generate raw hint text. Returns (text, from_gemini)."""
    if help_level == HelpLevel.NONE:
        return "", False

    key = _get_api_key(api_key)
    if not key:
        logger.warning("Falling back to template hint: GEMINI_API_KEY is not set.")
        return _offline_hint(code, error, help_level) or "", False

    try:
        from google import genai

        client = genai.Client(api_key=key)
        prompt = _build_hint_prompt(code, error, help_level, reference_solution_summary)
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text
        if text:
            return text.strip(), True
        logger.warning("Falling back to template hint: Gemini response was empty.")
        return _offline_hint(code, error, help_level) or "", False
    except Exception as err:
        logger.warning("Falling back to template hint due to %s: %s", type(err).__name__, err)
        return _offline_hint(code, error, help_level) or "", False


async def write_hint_async(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    jev: JevClient | None = None,
) -> HintResult:
    """Generate a pedagogical hint using Gemini with Jev leak verification.

    Args:
        code: Learner's current code.
        error: Current error or failing test assertion.
        help_level: HelpLevel agreed upon by the agents.
        reference_solution_summary: Summary of the task's reference solution.
        api_key: Optional Gemini API key override.
        model: Gemini model identifier.
        jev: Optional JevClient for verifying hints against reference solution leaks.
    """
    if help_level == HelpLevel.NONE:
        return HintResult(text=None, level=HelpLevel.NONE, verified=False, downgraded=False)

    candidate, _ = await _generate_raw_hint(
        code=code,
        error=error,
        help_level=help_level,
        reference_solution_summary=reference_solution_summary,
        api_key=api_key,
        model=model,
    )

    # If Jev and a reference summary are both given, check hint for leaks
    if jev is not None and reference_solution_summary:
        leak_question = (
            "Does `hint` reveal the approach in `reference_solution_summary` closely enough "
            "that the learner could write the solution without further thinking?"
        )
        leak_state = {"hint": candidate, "reference_solution_summary": reference_solution_summary}
        leak_check = await jev.noul(leak_state, leak_question)

        if leak_check.probability <= 0.5:
            return HintResult(
                text=candidate,
                level=help_level,
                verified=True,
                downgraded=False,
            )

        # Leak detected (> 0.5). Try generating again one level lower (stop at level 1)
        if help_level > HelpLevel.NUDGE:
            lower_level = help_level - 1
            lower_candidate, _ = await _generate_raw_hint(
                code=code,
                error=error,
                help_level=lower_level,
                reference_solution_summary=reference_solution_summary,
                api_key=api_key,
                model=model,
            )
            lower_state = {"hint": lower_candidate, "reference_solution_summary": reference_solution_summary}
            lower_check = await jev.noul(lower_state, leak_question)
            if lower_check.probability <= 0.5:
                return HintResult(
                    text=lower_candidate,
                    level=lower_level,
                    verified=True,
                    downgraded=True,
                )

        # If that also fails (or help_level was 1), return template hint for the original level
        template_hint = _offline_hint(code, error, help_level)
        return HintResult(
            text=template_hint,
            level=help_level,
            verified=False,
            downgraded=True,
        )

    return HintResult(
        text=candidate,
        level=help_level,
        verified=False,
        downgraded=False,
    )


async def explain_deliberation_async(
    decision: Decision,
    slices: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate an explanation of the multi-agent deliberation using Gemini API (async)."""
    key = _get_api_key(api_key)
    if not key:
        logger.warning("Falling back to template explanation: GEMINI_API_KEY is not set.")
        return _offline_explanation(decision, slices)

    try:
        from google import genai

        client = genai.Client(api_key=key)
        prompt = _build_explainer_prompt(decision, slices)
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text
        if text:
            return text.strip()
        logger.warning("Falling back to template explanation: Gemini response was empty.")
        return _offline_explanation(decision, slices)
    except Exception as err:
        logger.warning("Falling back to template explanation due to %s: %s", type(err).__name__, err)
        return _offline_explanation(decision, slices)


def write_hint(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    jev: JevClient | None = None,
) -> HintResult:
    """Synchronous wrapper for write_hint_async."""
    import asyncio

    return asyncio.run(
        write_hint_async(
            code=code,
            error=error,
            help_level=help_level,
            reference_solution_summary=reference_solution_summary,
            api_key=api_key,
            model=model,
            jev=jev,
        )
    )


def explain_deliberation(
    decision: Decision,
    slices: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Synchronous wrapper for explain_deliberation_async."""
    import asyncio

    return asyncio.run(
        explain_deliberation_async(
            decision=decision,
            slices=slices,
            api_key=api_key,
            model=model,
        )
    )
