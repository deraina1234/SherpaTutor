"""LLM Hint Writer & Deliberation Explainer for Sherpa.

Per Section 10 of the Spotter design doc:
- Hint writer: Receives code, error, agreed level, and reference solution policy.
  Generates a pedagogical hint strictly adhering to the agreed level (Levels 0-4).
- Hint checking: When Jev is provided, it checks every hint twice: that it claims nothing
  the code and test results don't show, and (given a reference solution summary) that it
  does not leak the solution. A rejected hint is rewritten at the same level with the
  rejected draft and the reason, then one level lower, then falls back to template.
- Explainer: Code writes the first sentence (the test result) and the last (what Sherpa
  does, at the level actually given). The LLM writes only the "why" in between, from a
  list of deliberation facts; Jev checks it against them, else a template is used.
- The LLM never decides whether to intervene; it only executes after the multi-agent
  deliberation finishes. During deliberation, complete_text() only words the agents'
  arguments (see agents.py); Jev checks those words and decides what they are worth.

Providers, tried in order: Nemotron (any OpenAI-compatible endpoint, e.g. NVIDIA's hosted
API or vLLM on a Brev GPU; enabled by NEMOTRON_BASE_URL + NEMOTRON_MODEL), then Google
Gemini (google-genai SDK), then deterministic offline templates for testing and demo
reliability.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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
# Nemotron (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------

NEMOTRON_TIMEOUT_SECONDS = 10.0  # per attempt; replies normally take about a second
NEMOTRON_ATTEMPTS = 2  # one retry on a timeout, connection error, 429, or 5xx
NEMOTRON_MAX_TOKENS = 400
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def nemotron_enabled() -> bool:
    return bool(os.getenv("NEMOTRON_BASE_URL") and os.getenv("NEMOTRON_MODEL"))


async def _nemotron_complete(prompt: str, transport: Any = None) -> str | None:
    """One chat completion from the configured Nemotron endpoint, or None on any failure.

    Env: NEMOTRON_BASE_URL (ending in /v1), NEMOTRON_MODEL, optional NEMOTRON_API_KEY,
    optional NEMOTRON_SYSTEM (e.g. "/no_think" for models that switch reasoning by system
    prompt), optional NEMOTRON_ENABLE_THINKING ("true" to allow reasoning; off by default,
    since hints should be short and fast). Reasoning text is never shown: only `content`
    is read, and any <think>...</think> block in it is stripped.
    """
    if not nemotron_enabled():
        return None
    import httpx

    base = os.environ["NEMOTRON_BASE_URL"].rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.getenv("NEMOTRON_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['NEMOTRON_API_KEY']}"
    messages = [{"role": "user", "content": prompt}]
    if os.getenv("NEMOTRON_SYSTEM"):
        messages.insert(0, {"role": "system", "content": os.environ["NEMOTRON_SYSTEM"]})
    body = {
        "model": os.environ["NEMOTRON_MODEL"],
        "messages": messages,
        "max_tokens": NEMOTRON_MAX_TOKENS,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": os.getenv("NEMOTRON_ENABLE_THINKING", "").lower() == "true"},
    }
    content = ""
    async with httpx.AsyncClient(timeout=NEMOTRON_TIMEOUT_SECONDS, transport=transport) as client:
        for attempt in range(1, NEMOTRON_ATTEMPTS + 1):
            try:
                response = await client.post(f"{base}/chat/completions", json=body, headers=headers)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
                break
            except Exception as err:
                status = err.response.status_code if isinstance(err, httpx.HTTPStatusError) else None
                transient = status is None or status == 429 or status >= 500
                if transient and attempt < NEMOTRON_ATTEMPTS:
                    logger.warning("Nemotron request failed (%s: %s); retrying.", type(err).__name__, err)
                    continue
                logger.warning("Nemotron request failed (%s: %s); trying the next provider.", type(err).__name__, err)
                return None
    text = _THINK_BLOCK.sub("", content).strip()
    if not text:
        logger.warning("Nemotron returned no text; trying the next provider.")
        return None
    return text


async def complete_text(prompt: str, api_key: str | None = None, model: str = DEFAULT_GEMINI_MODEL) -> str | None:
    """Plain completion for agents.Writer: Nemotron, then Gemini, else None (no template)."""
    text = await _nemotron_complete(prompt)
    if text:
        return text
    key = _get_api_key(api_key)
    if not key:
        return None
    try:
        from google import genai

        response = await genai.Client(api_key=key).aio.models.generate_content(model=model, contents=prompt)
        return (response.text or "").strip() or None
    except Exception as err:
        logger.warning("Gemini completion failed (%s: %s).", type(err).__name__, err)
        return None


# ---------------------------------------------------------------------------
# Offline Template Fallbacks (for offline testing & emergency demo backup)
# ---------------------------------------------------------------------------


ALL_TESTS_PASS_TEXT = (
    "All test cases pass. If you want to go further, try an edge case of your own "
    "(an empty input, one element, or the largest input) or tidy up a line you find hard to read."
)


def _offline_hint(code: str, error: str | None, help_level: int, last_passed: bool | None = None) -> str | None:
    """Generate a high-quality deterministic hint when offline."""
    if help_level == HelpLevel.NONE:
        return None
    if last_passed:
        return ALL_TESTS_PASS_TEXT

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


# ---------------------------------------------------------------------------
# Core LLM Functions
# ---------------------------------------------------------------------------


def _run_result_text(error: str | None, last_passed: bool | None) -> str:
    if last_passed:
        return (
            "TEST RESULTS:\nAll test cases pass. The code is correct, so do not claim it has a bug. "
            "At most, suggest an edge case to double-check or a readability improvement."
        )
    if error:
        return f"FIRST FAILING TEST / ERROR:\n{error}"
    if last_passed is False:
        return "TEST RESULTS:\nThe last run failed, but no error message was captured."
    return "TEST RESULTS:\nThe learner has not run their code yet."


def _build_hint_prompt(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    last_passed: bool | None = None,
    rejected_draft: str | None = None,
    problem_statement: str | None = None,
    rejected_reason: str = "leak",
) -> str:
    level_instructions = {
        HelpLevel.NUDGE: (
            "Goal: Write a short NUDGE (1-2 sentences) about their APPROACH.\n"
            "- Ask one guiding question about their strategy: what it costs, where it will struggle, or which "
            "property of the problem it does not use yet. A failing test result can serve as evidence.\n"
            "- DO NOT name the technique, data structure, or algorithm they should switch to, and DO NOT describe the fix.\n"
            "- DO NOT write or suggest code.\n"
            "- Ground the question in this student's code and this problem's input and output.\n"
            "- If their approach is sound and only a detail is wrong, do not explain the detail or mention "
            "where in the code it is. Use this shape, filled in from the test result: \"Your function returns "
            "<wrong result> for <input>. Walk through that input by hand: where does it stop doing what you expect?\"\n"
            "Example of a bad nudge (names the technique): \"Use a sliding window with a set of characters.\""
        ),
        HelpLevel.CONCEPTUAL_HINT: (
            "Goal: Write a CONCEPTUAL HINT (1-3 sentences) about their APPROACH.\n"
            "- Explain the idea their approach is missing and why it fits this problem, e.g. what a property of "
            "the input (sorted order, a running total, a bound) would let them skip or reuse.\n"
            "- Use plain English. DO NOT write code, pseudo-code, or a step-by-step algorithm, or give line numbers."
        ),
        HelpLevel.POINT_TO_AREA: (
            "Goal: POINT TO THE AREA (1-3 sentences).\n"
            "- Point out the specific loop, line, or conditional where their approach goes wrong or wastes work.\n"
            "- Explain what that part does vs what the approach needs it to do.\n"
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
    if rejected_draft and rejected_reason == "ungrounded":
        reference_constraint += (
            f"\nA previous draft claimed something the code and test results do not show and was rejected:\n"
            f"\"{rejected_draft}\"\n"
            "Write a new hint that states only what the code and test results below actually show."
        )
    elif rejected_draft:
        reference_constraint += (
            f"\nA previous draft gave away too much of the fix and was rejected:\n\"{rejected_draft}\"\n"
            "Write a new hint that reveals less: ask a question about their approach, "
            "without naming the technique, the missing check, the step to add, or the fix."
        )

    problem = f"PROBLEM:\n{problem_statement}\n\n" if problem_statement else ""
    return (
        "You are an encouraging, pedagogical AI programming tutor helping a student.\n"
        "Hints are about the student's APPROACH: first judge whether their overall strategy can solve the "
        "problem correctly and efficiently, then aim the hint at the idea they are missing. Mention a small "
        "mechanical slip only if the approach is already sound.\n"
        "If the code looks unfinished and has not been run (for example no return statement yet), the student "
        "is still typing: do not point out what is missing; talk about the approach.\n"
        f"{instructions}\n"
        f"{reference_constraint}\n\n"
        f"{problem}"
        f"STUDENT CODE:\n```python\n{code}\n```\n\n"
        f"{_run_result_text(error, last_passed)}\n\n"
        "Only claim a bug you can point to in the code or the test results above. "
        "Never invent a bug (such as an indentation or syntax problem) that is not there.\n"
        "Respond with ONLY the hint text. Keep it concise, friendly, and pedagogically sound."
    )


_LEVEL_NAMES = ["no help", "a nudge", "a conceptual hint", "pointing to the problem area", "a near-solution"]

# REASON_TAG -> plain-language reason, for the explainer prompt.
_CLAIM_TEXT = {
    "concept_gap": "you seem to be missing a concept, not just making a slip",
    "repeated_error": "the same error keeps coming back",
    "long_idle": "you had stopped editing for a while",
    "learner_requested_help": "you asked for help",
    "active_progress": "you were making progress on your own",
    "recent_hint_given": "you had just received a hint",
    "solution_leak_risk": "pointing at the problem would give away the solution",
}

_STOP_RULES = {
    OutcomeKind.CONSENSUS: "Learning and Momentum voted the same level, so Sherpa acts at that level.",
    OutcomeKind.SOFT_DISAGREEMENT: "Learning and Momentum ended one level apart, so Sherpa takes the lower, less intrusive level.",
    OutcomeKind.DEADLOCK: (
        "Learning and Momentum ended two or more levels apart, so Sherpa gives no automatic hint "
        "and lets the student ask for a nudge instead."
    ),
}


def _level_name(level: int) -> str:
    return _LEVEL_NAMES[level] if 0 <= level < len(_LEVEL_NAMES) else f"level {level}"


def _deliberation_facts(decision: Decision) -> list[str]:
    """One plain-language line per fact the explainer may use, from round 0 to the end."""
    facts: list[str] = []
    first = {p.agent: p for p in decision.rounds[0]} if decision.rounds else {}
    last = {p.agent: p for p in decision.rounds[-1]} if decision.rounds else {}
    for agent, position in first.items():
        # Signals are what the agent noticed, not necessarily why it voted as it did: an agent
        # can notice "you asked for help" and still vote for no help because the tests pass.
        signals = [_CLAIM_TEXT.get(a.claim, a.claim) for a in position.arguments if "responds_to" not in a.evidence]
        noticed = f" Signals it noticed: {'; '.join(signals)}." if signals else ""
        if position.cap is not None:
            facts.append(f"Integrity set a ceiling of {_level_name(position.cap)} (it does not vote).{noticed}")
        else:
            facts.append(f"{agent.title()} first voted for {_level_name(position.level)}, judging from your code and test results.{noticed}")
        for a in position.arguments:
            if a.text and "responds_to" not in a.evidence:
                facts.append(f'{agent.title()} argued: "{a.text}"')
    for agent, position in last.items():
        for a in position.arguments:
            target = a.evidence.get("responds_to")
            if target is None:
                continue
            peer = str(target).split(".", 1)[0].title()
            verdict = "agreed with" if a.attacks is None else "rejected"
            said = f' It said: "{a.text}"' if a.text else ""
            facts.append(
                f"In round {a.round}, {agent.title()} {verdict} {peer}'s point that "
                f"{_CLAIM_TEXT.get(a.claim, a.claim)}.{said}"
            )
        if position.cap is None and agent in first and position.level != first[agent].level:
            facts.append(f"{agent.title()} moved from {_level_name(first[agent].level)} to {_level_name(position.level)}.")
    return facts


def _build_explainer_prompt(decision: Decision, slices: dict[str, Any] | None = None) -> str:
    last_run = ((slices or {}).get("learning") or {}).get("last_run") or {}
    passed = bool(last_run.get("passed"))
    if passed:
        run_text = "The student's latest run passed every test."
    elif last_run.get("error"):
        run_text = f"The student's latest run failed: {last_run['error']}"
    elif last_run.get("passed") is False:
        run_text = "The student's latest run failed."
    else:
        run_text = "The student has not run their code yet."

    if decision.final_level is None:
        result = "No automatic hint; the student can ask for a nudge."
    else:
        result = f"Sherpa gives {_level_name(decision.final_level)}."
    cap_text = (
        "A vote above Integrity's ceiling was lowered to that ceiling."
        if decision.cap_applied
        else "No vote was above Integrity's ceiling, so the ceiling changed nothing."
    )
    facts = "\n".join(f"- {fact}" for fact in _deliberation_facts(decision))

    return (
        "You are the explainer component of Sherpa, a multi-agent coding tutor.\n"
        "Three agents decided how much help to give a student:\n"
        "- Learning: protects long-term learning and productive struggle.\n"
        "- Momentum: steps in before the student stalls out.\n"
        "- Integrity: sets a ceiling so hints never reveal the solution. It does not vote.\n\n"
        f"{run_text}\n\n"
        f"What happened:\n{facts}\n\n"
        f"Rule applied: {_STOP_RULES[decision.outcome]}\n"
        f"{cap_text}\n"
        f"Result: {result}\n\n"
        "Write 1-2 plain, friendly sentences to the student (\"you\") explaining WHY the agents landed here. "
        "Do not restate the test result or the Result: Sherpa writes those sentences itself, before and after yours. "
        + (
            "Every test passed. "
            if passed
            else "The code still fails, so never say the tests passed or that no help was needed. "
        )
        + "Use only the facts above. Do not invent reasons, agreements, feelings, or a cap that was not applied. "
        "Do not claim an agent voted a level because of a signal that points the other way "
        "(for example, no help because you asked for help), "
        "and do not describe Integrity as voting or siding with anyone."
    )


async def _generate_raw_hint(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None,
    api_key: str | None,
    model: str,
    last_passed: bool | None = None,
    rejected_draft: str | None = None,
    problem_statement: str | None = None,
    rejected_reason: str = "leak",
) -> tuple[str, bool]:
    """Generate raw hint text. Returns (text, from_llm)."""
    if help_level == HelpLevel.NONE:
        return "", False

    prompt = _build_hint_prompt(
        code, error, help_level, reference_solution_summary, last_passed, rejected_draft, problem_statement, rejected_reason
    )
    nemotron_text = await _nemotron_complete(prompt)
    if nemotron_text:
        logger.info("Hint L%d written by Nemotron.", help_level)
        return nemotron_text, True

    key = _get_api_key(api_key)
    if not key:
        logger.warning("Falling back to template hint: GEMINI_API_KEY is not set.")
        return _offline_hint(code, error, help_level, last_passed) or "", False

    try:
        from google import genai

        client = genai.Client(api_key=key)
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = response.text
        if text:
            logger.info("Hint L%d written by Gemini.", help_level)
            return text.strip(), True
        logger.warning("Falling back to template hint: Gemini response was empty.")
        return _offline_hint(code, error, help_level, last_passed) or "", False
    except Exception as err:
        logger.warning("Falling back to template hint due to %s: %s", type(err).__name__, err)
        return _offline_hint(code, error, help_level, last_passed) or "", False


async def write_hint_async(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    jev: JevClient | None = None,
    last_passed: bool | None = None,
    problem_statement: str | None = None,
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
        last_passed: Whether the learner's last run passed (None if never run), so the
            writer never invents a bug in passing code.
        problem_statement: The problem as the learner sees it, so the hint can judge their approach.
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
        last_passed=last_passed,
        problem_statement=problem_statement,
    )

    if jev is None:
        return HintResult(text=candidate, level=help_level, verified=False, downgraded=False)

    async def problem_with(hint: str) -> str | None:
        return await _hint_problem(jev, hint, code, error, last_passed, problem_statement, reference_solution_summary)

    problem = await problem_with(candidate)
    if problem is None:
        return HintResult(text=candidate, level=help_level, verified=True, downgraded=False)

    # Rejected. Rewrite at the same level with the rejected draft and the reason as a
    # counterexample, then one level lower (stopping at a nudge), then fall back to the template.
    retries = [help_level] + ([help_level - 1] if help_level > HelpLevel.NUDGE else [])
    rejected = candidate
    for level in retries:
        logger.info("Hint L%d draft rejected (%s); rewriting at L%d: %s", help_level, problem, level, rejected)
        retry, _ = await _generate_raw_hint(
            code=code,
            error=error,
            help_level=level,
            reference_solution_summary=reference_solution_summary,
            api_key=api_key,
            model=model,
            last_passed=last_passed,
            rejected_draft=rejected,
            rejected_reason=problem,
            problem_statement=problem_statement,
        )
        if not retry:
            continue
        retry_problem = await problem_with(retry)
        if retry_problem is None:
            return HintResult(text=retry, level=level, verified=True, downgraded=level < help_level)
        rejected, problem = retry, retry_problem

    logger.info("Every hint draft was rejected (last: %s, %s); using the template hint.", problem, rejected)
    return HintResult(
        text=_offline_hint(code, error, help_level, last_passed),
        level=help_level,
        verified=False,
        downgraded=True,
    )


_LEAK_QUESTION = (
    "Does `hint` reveal the approach in `reference_solution_summary` closely enough "
    "that the learner could write the solution without further thinking?"
)
LEAK_PROBABILITY = 0.5  # a hint judged above this leaks the solution

_UNGROUNDED_QUESTION = (
    "Does `hint` claim something about the student's code or its results that `code` and "
    "`run_result` do not show, such as a bug, an output value, an error, or a line that is not "
    "there? Judgments about the approach's speed or strategy are not claims of this kind."
)
UNGROUNDED_PROBABILITY = 0.5  # a hint judged above this states something the code and results don't show

# Hint rejection reasons, also used to word the rewrite request in _build_hint_prompt.
LEAK = "leak"
UNGROUNDED = "ungrounded"


async def _hint_problem(
    jev: JevClient,
    hint: str,
    code: str,
    error: str | None,
    last_passed: bool | None,
    problem_statement: str | None,
    reference_solution_summary: str | None,
) -> str | None:
    """LEAK, UNGROUNDED, or None when Jev finds the hint both safe and true to the code and results."""
    grounding_state = {
        "hint": hint,
        "problem": problem_statement or "",
        "code": code,
        "run_result": _run_result_text(error, last_passed),
    }
    checks = [jev.noul(grounding_state, _UNGROUNDED_QUESTION)]
    if reference_solution_summary:
        checks.append(jev.noul({"hint": hint, "reference_solution_summary": reference_solution_summary}, _LEAK_QUESTION))
    results = await asyncio.gather(*checks)
    if len(results) > 1 and results[1].probability > LEAK_PROBABILITY:
        return LEAK
    if results[0].probability > UNGROUNDED_PROBABILITY:
        return UNGROUNDED
    return None


_WHY_CHECK_QUESTION = (
    "Does `explanation` state anything that `facts` do not support, or contradict `run_result`, "
    "`rule`, or `result`?"
)
WHY_CHECK_PROBABILITY = 0.5  # an explanation judged above this is replaced by the template


def _last_run(slices: dict[str, Any] | None) -> dict[str, Any]:
    return ((slices or {}).get("learning") or {}).get("last_run") or {}


def _opening(slices: dict[str, Any] | None) -> str:
    """First sentence, written by code: the test result."""
    if slices is None:
        return ""
    last_run = _last_run(slices)
    if last_run.get("passed"):
        return "Your tests all pass."
    if last_run.get("error"):
        return f"Your latest run failed: {str(last_run['error']).rstrip('.')}."
    if last_run.get("passed") is False:
        return "Your latest run failed."
    return "You haven't run your code yet."


def _closing(decision: Decision, given_level: int | None = None) -> str:
    """Last sentence, written by code: what Sherpa actually does, at the level actually given."""
    if decision.final_level is None:
        return "So there is no automatic hint; ask for a nudge whenever you want one."
    level = decision.final_level if given_level is None else given_level
    if level == 0:
        return "So Sherpa is holding back for now; you can ask for help any time."
    if level < decision.final_level:
        return (
            f"So Sherpa is giving you {_level_name(level)}, a step lighter than the agents chose, "
            "because the stronger hint did not pass Sherpa's checks."
        )
    return f"So Sherpa is giving you {_level_name(level)}."


def _join(items: list[str]) -> str:
    # Semicolons, since an item may contain a comma ("a concept, not just making a slip").
    return items[0] if len(items) == 1 else f"{'; '.join(items[:-1])}; and {items[-1]}"


def _template_why(decision: Decision) -> str:
    """The middle of the explanation without an LLM: what the agents noticed and the rule applied."""
    signals: list[str] = []
    for position in decision.rounds[0] if decision.rounds else []:
        if position.cap is not None:
            continue
        for tag in position.reason_tags:
            text = _CLAIM_TEXT.get(tag)
            if text and text not in signals:
                signals.append(text)
    parts = [f"The agents noticed that {_join(signals)}."] if signals else []
    parts.append(_STOP_RULES[decision.outcome])
    if decision.cap_applied:
        parts.append("Integrity's ceiling lowered a vote that asked for more help.")
    return " ".join(parts)


async def explain_why_async(
    decision: Decision,
    slices: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    jev: JevClient | None = None,
) -> str:
    """The "why" sentences: an LLM's, if Jev finds them true to the facts, else the template's."""
    why = await complete_text(_build_explainer_prompt(decision, slices), api_key, model)
    if not why:
        logger.warning("Falling back to template explanation: no LLM answered.")
        return _template_why(decision)
    if jev is not None:
        state = {
            "explanation": why,
            "run_result": _opening(slices) or "unknown",
            "facts": _deliberation_facts(decision),
            "rule": _STOP_RULES[decision.outcome],
            "result": _closing(decision),
        }
        check = await jev.noul(state, _WHY_CHECK_QUESTION)
        if check.probability > WHY_CHECK_PROBABILITY:
            logger.info("Explanation rejected by Jev (%.2f); using the template: %s", check.probability, why)
            return _template_why(decision)
    logger.info("Explanation written by the LLM.")
    return why


def frame_explanation(
    why: str, decision: Decision, slices: dict[str, Any] | None = None, given_level: int | None = None
) -> str:
    """Code-written test result, then the why, then the code-written outcome at the level given."""
    return " ".join(part for part in (_opening(slices), why.strip(), _closing(decision, given_level)) if part)


async def explain_deliberation_async(
    decision: Decision,
    slices: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    jev: JevClient | None = None,
    given_level: int | None = None,
) -> str:
    """Explain the deliberation in plain language. See explain_why_async and frame_explanation."""
    why = await explain_why_async(decision, slices, api_key, model, jev)
    return frame_explanation(why, decision, slices, given_level)


def write_hint(
    code: str,
    error: str | None,
    help_level: int,
    reference_solution_summary: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
    jev: JevClient | None = None,
    last_passed: bool | None = None,
    problem_statement: str | None = None,
) -> HintResult:
    """Synchronous wrapper for write_hint_async."""
    return asyncio.run(
        write_hint_async(
            code=code,
            error=error,
            help_level=help_level,
            reference_solution_summary=reference_solution_summary,
            api_key=api_key,
            model=model,
            jev=jev,
            last_passed=last_passed,
            problem_statement=problem_statement,
        )
    )


def explain_deliberation(
    decision: Decision,
    slices: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Synchronous wrapper for explain_deliberation_async."""
    return asyncio.run(
        explain_deliberation_async(
            decision=decision,
            slices=slices,
            api_key=api_key,
            model=model,
        )
    )
