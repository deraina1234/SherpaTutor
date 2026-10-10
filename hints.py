"""LLM Hint Writer & Deliberation Explainer for Sherpa.

Per Section 10 of the Spotter design doc:
- Hint writer: Receives code, error, agreed level, and reference solution policy.
  Generates a pedagogical hint strictly adhering to the agreed level (Levels 0-4).
- Explainer: Turns structured deliberation rounds, reason tags, and decisions into
  2-3 plain-language sentences for the side panel.
- The LLM never decides whether to intervene; it only executes after the multi-agent
  deliberation finishes.

Powered by Google Gemini (google-genai SDK), with deterministic offline fallback
templates for testing and hackathon demo reliability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from contracts import Decision, HelpLevel, OutcomeKind

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


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
    return explicit_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


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

    return (
        "You are the explainer component of Sherpa, a multi-agent coding tutor.\n"
        "Three peer agents with competing goals just deliberated on how much help to give a student:\n"
        "- Learning: protects long-term learning and productive struggle.\n"
        "- Momentum: minimizes frustration and steps in before stalling out.\n"
        "- Integrity: ensures hints never reveal the core task solution.\n\n"
        f"Outcome: {decision.outcome.value}\n"
        f"Final Help Level: {decision.final_level}\n"
        f"Integrity Cap Applied: {decision.cap_applied}\n"
        f"Ask Learner (Deadlock): {decision.ask_learner}\n\n"
        f"Deliberation History:\n{rounds_text}\n\n"
        "Write 2-3 plain-language, encouraging sentences explaining to the student why the agents took "
        "their stances and why this final outcome was chosen. "
        "Highlight any conflict, consensus, or cap naturally without sounding overly robotic."
    )


async def write_hint_async(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str | None:
    """Generate a hint using Gemini API (async). Falls back to template if offline."""
    if help_level == HelpLevel.NONE:
        return None

    key = _get_api_key(api_key)
    if not key:
        return _offline_hint(code, error, help_level)

    try:
        from google import genai

        client = genai.Client(api_key=key)
        prompt = _build_hint_prompt(code, error, help_level, reference_solution_summary)
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text
        return text.strip() if text else _offline_hint(code, error, help_level)
    except Exception:
        # Fallback cleanly on network/API failure
        return _offline_hint(code, error, help_level)


async def explain_deliberation_async(
    decision: Decision,
    slices: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Generate an explanation of the multi-agent deliberation using Gemini API (async)."""
    key = _get_api_key(api_key)
    if not key:
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
        return text.strip() if text else _offline_explanation(decision, slices)
    except Exception:
        return _offline_explanation(decision, slices)


def write_hint(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str | None:
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
