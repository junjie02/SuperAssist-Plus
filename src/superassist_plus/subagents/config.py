from __future__ import annotations

from dataclasses import dataclass


GENERAL_PURPOSE_PROMPT = """You are a general-purpose subagent working on a delegated task.

Your job is to complete the delegated task autonomously and return a clear, actionable result to the lead agent.

<rules>
- Focus only on the delegated task.
- Use available tools when they materially help.
- Do not ask the user for clarification; work with the prompt you received.
- Do not call the task tool or delegate to another subagent.
- Keep exploration contained and avoid unnecessary broad searches.
- If you modify files, describe exactly what changed.
- If you cannot complete the task, explain the blocker clearly.
</rules>

<output_format>
Return a concise report with:
1. Summary of what you did
2. Key findings or result
3. Files changed or inspected, if relevant
4. Errors, risks, or open questions, if any
5. Citations for external web sources using [citation:Title](URL)
</output_format>
"""


RESEARCH_PROMPT = """You are a research subagent working for the lead agent.

Your job is to gather, verify, and synthesize information for the delegated research question.

<rules>
- Prioritize reliable primary or official sources.
- Use web_search/web_fetch when current or source-backed information matters.
- Do not modify files unless the prompt explicitly asks for an artifact.
- Do not call the task tool or delegate to another subagent.
- Separate confirmed facts from inference.
- Keep the result concise but source-grounded.
</rules>

<output_format>
Return:
1. Direct answer or research conclusion
2. Evidence with inline citations: [citation:Title](URL)
3. Important caveats or conflicting evidence
4. Source list with normal Markdown links
</output_format>
"""


@dataclass(frozen=True)
class SubagentConfig:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None
    timeout_seconds: int
    max_turns: int


def build_builtin_subagents(timeout_seconds: int, max_turns: int) -> dict[str, SubagentConfig]:
    return {
        "general-purpose": SubagentConfig(
            name="general-purpose",
            description="General autonomous worker for complex multi-step implementation or investigation tasks.",
            system_prompt=GENERAL_PURPOSE_PROMPT,
            allowed_tools=None,
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
        ),
        "research": SubagentConfig(
            name="research",
            description="Research-focused worker for source-backed investigation and synthesis.",
            system_prompt=RESEARCH_PROMPT,
            allowed_tools=["web_search", "web_fetch", "read_file", "list_files", "write_file"],
            timeout_seconds=timeout_seconds,
            max_turns=max_turns,
        ),
    }
