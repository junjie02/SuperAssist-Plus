"""Prompt assembly for the SuperAssist memory writer.

Ported from CogniFold's prompts.py. Composes sections from prompt_sections.py
based on domain configuration and reasoning mode.
"""

from __future__ import annotations

from superassist.memory.domain import DomainConfig
from superassist.memory.prompt_sections import (
    DEFAULT_SECTION_ORDER,
    SECTION_REGISTRY,
)


def resolve_sections(
    disabled_sections: frozenset[str] = frozenset(),
    extra_sections: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve which sections to include, in order."""
    result: list[tuple[str, str]] = []
    for key in DEFAULT_SECTION_ORDER:
        if key in disabled_sections:
            continue
        content = SECTION_REGISTRY.get(key)
        if content:
            result.append((key, content))

    for key, content in (extra_sections or {}).items():
        result.append((key, content))

    return result


# ---------------------------------------------------------------------------
# Reasoning modes
# ---------------------------------------------------------------------------

MODE_QUICK = """

## Reasoning Mode: QUICK

- 必须创建 1 个 event 节点 (ref="current_event") 概括本次对话
- 提取 0-2 个新概念 (只提取明确出现的新知识点)
- 每个概念加 1 条 GROUNDS 边连到 event
- 每回合 3-6 个操作即可 (event + 边 = 至少 1 ADD_NODE + 1 ADD_EDGE)
- 优先更新已有概念, 不要创建重复概念
"""

MODE_ANALYTICAL = """

## Reasoning Mode: ANALYTICAL

In analytical mode, perform deep analysis:
- Examine patterns across the memory context
- Look for hidden connections and emerging themes
- Consider creating hierarchical concepts (level 1/2/3)
- Analyze concept importance changes carefully
- Be thorough in your reasoning
"""

MODE_CONSOLIDATION = """

## Reasoning Mode: CONSOLIDATION

In consolidation mode, focus on graph health:
- Identify similar or duplicate concepts that should be merged
- Look for weak concepts (importance < 0.3) that could be removed
- Find orphan nodes that need connections
- Create parent concepts for clusters of related concepts
- Prefer MERGE_NODES and REMOVE_EDGE operations
"""

MODE_PROMPTS = {
    "quick": MODE_QUICK,
    "analytical": MODE_ANALYTICAL,
    "consolidation": MODE_CONSOLIDATION,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_memory_writer_prompt(
    domain: DomainConfig | None = None,
    mode: str = "quick",
) -> str:
    """Format the memory writer system prompt.

    Args:
        domain: Domain configuration. Uses LEARNING_WIKI_DOMAIN if None.
        mode: Reasoning mode: "quick", "analytical", or "consolidation".

    Returns:
        Complete system prompt string.
    """
    if domain is None:
        from superassist.memory.domain import LEARNING_WIKI_DOMAIN

        domain = LEARNING_WIKI_DOMAIN

    sections = resolve_sections(
        disabled_sections=domain.disabled_sections,
        extra_sections=domain.extra_sections,
    )

    parts: list[str] = []
    for _key, content in sections:
        parts.append(content)

    prompt = "\n\n".join(parts)

    if mode in MODE_PROMPTS:
        prompt += MODE_PROMPTS[mode]

    return prompt
