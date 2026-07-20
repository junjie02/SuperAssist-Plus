"""Domain configuration for SuperAssist memory writer.

Ported from CogniFold's domain.py. Each domain defines the node/edge semantics,
concept/intent guidelines, and section composition for a specific use case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainConfig:
    """Configuration for a memory writer domain.

    Attributes:
        name: Domain identifier.
        description: What this domain represents.
        event_description: How events are described in this domain.
        node_type_descriptions: Per-node-type descriptions for the prompt.
        concept_guidelines: Domain-specific concept discovery rules.
        intent_guidelines: Domain-specific intent discovery rules.
        disabled_sections: Section keys to exclude from the prompt.
        extra_sections: Custom sections to inject.
    """

    name: str
    description: str
    event_description: str
    node_type_descriptions: dict[str, str] = field(default_factory=dict)
    concept_guidelines: tuple[str, ...] = field(default_factory=tuple)
    intent_guidelines: tuple[str, ...] = field(default_factory=tuple)
    disabled_sections: frozenset[str] = field(default_factory=frozenset)
    extra_sections: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Learning-Wiki Domain
# ---------------------------------------------------------------------------

LEARNING_WIKI_DOMAIN = DomainConfig(
    name="learning-wiki",
    description="从飞书消息中提取的学习笔记、文档片段和知识问答, 构建个人知识图谱",
    event_description="飞书消息中的学习内容 (笔记、文档片段、问答)",
    node_type_descriptions={
        "event": "一条学习相关的消息或文档片段 (已存在, 不需要创建)",
        "concept": "关键知识点、术语定义、概念之间的关系",
        "intent": "学习目标 (复习某主题、做练习题、查阅资料)",
        "time": "截止日期、学习计划时间锚点",
    },
    concept_guidelines=(
        "提取关键术语、定义和核心概念",
        "识别概念之间的因果关系和层级关系",
        "当多个消息涉及同一主题时创建或强化概念",
        "区分事实性知识和用户的个人理解/笔记",
        "记录用户的偏好和兴趣领域",
        "跟踪学习进度 (哪些主题已经学过, 哪些还需要深入)",
    ),
    intent_guidelines=(
        "检测知识缺口, 创建 '进一步学习' 意图",
        "当概念集群形成时 (3+ 相关概念), 创建 '复习' 意图",
        "当用户明确提到后续计划时, 创建时间绑定的学习意图",
        "如有明确截止日期, 创建关联 TIME 节点的意图",
    ),
    disabled_sections=frozenset(),
    extra_sections={},
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "learning-wiki": LEARNING_WIKI_DOMAIN,
}


def get_domain(name: str) -> DomainConfig:
    """Get a domain configuration by name."""
    if name not in DOMAIN_REGISTRY:
        available = ", ".join(DOMAIN_REGISTRY.keys())
        raise KeyError(f"Unknown domain: {name}. Available: {available}")
    return DOMAIN_REGISTRY[name]
