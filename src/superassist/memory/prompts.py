"""Prompt assembly for the SuperAssist memory writer."""

from __future__ import annotations

from superassist.memory.prompt_sections import (
    DEFAULT_SECTION_ORDER,
    SECTION_REGISTRY,
)

MODE_QUICK = """

## Reasoning Mode: QUICK

- 有持久信息时创建 1 个 event 节点 (ref="current_event")；纯问候或无持久信息时返回空 operations
- 提取 0-2 个新概念 (只提取明确出现的新知识点)
- 每个概念加 1 条 GROUNDS 边连到 event
- 需要写入时通常使用 2-6 个操作；无需写入时不创建占位节点
- 优先更新已有概念, 不要创建重复概念
"""


def format_memory_writer_prompt() -> str:
    """Build the single prompt used by the memory writer."""

    sections = [SECTION_REGISTRY[key] for key in DEFAULT_SECTION_ORDER if SECTION_REGISTRY.get(key)]
    return "\n\n".join(sections) + MODE_QUICK
