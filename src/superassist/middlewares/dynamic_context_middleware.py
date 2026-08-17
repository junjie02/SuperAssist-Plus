"""Inject dynamic runtime context (memory recall and skills) into model calls.

This is the wrap_model_call counterpart to ProAssist's DynamicContextMiddleware.
The static system prompt is supplied via ``create_agent(system_prompt=...)``;
this middleware appends per-call context after the stable conversation prefix.
That placement lets provider prompt caching reuse the static prompt and history.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from superassist.agent.state import SuperAssistState
from superassist.skills import active_skill_names, build_loaded_skills_section


class DynamicContextMiddleware(AgentMiddleware[SuperAssistState]):
    """Append turn-time context without mutating the stable system prefix."""

    state_schema = SuperAssistState

    def __init__(
        self,
        skill_active_ttl_seconds: int = 300,
        *,
        clock: Callable[[], float] = time.time,
        preserve_static_prefix: bool = True,
        explicit_prompt_cache: bool = False,
        quiz_context_provider: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._skill_active_ttl_seconds = skill_active_ttl_seconds
        self._clock = clock
        self._preserve_static_prefix = preserve_static_prefix
        self._explicit_prompt_cache = explicit_prompt_cache
        self._quiz_context_provider = quiz_context_provider

    def before_model(self, state: SuperAssistState, runtime: Runtime) -> dict[str, Any] | None:
        metadata = dict(state.get("metadata") or {})
        metadata["dynamic_context_injected"] = True
        return {"metadata": metadata}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        state = request.state or {}
        memory_recall = state.get("memory_recall", {})
        currently_active = set(active_skill_names(
            state.get("skill_activations"),
            self._skill_active_ttl_seconds,
            now=self._clock(),
        ))
        loaded_skills = sorted(
            currently_active.intersection(state.get("active_skills_at_turn_start") or [])
        )
        user_id = state.get("user_id", "local-user")
        thread_id = state.get("thread_id", "")
        rag_mode = bool(state.get("rag_mode"))

        loaded_section = build_loaded_skills_section(loaded_skills)

        reminder_lines = [
            "<TurnContext>",
            "<RuntimeContext>",
            f"- user_id: {user_id}",
            f"- thread_id: {thread_id}",
            "</RuntimeContext>",
            '<LongTermMemory format="json">',
            json.dumps(memory_recall, ensure_ascii=False, separators=(",", ":")),
            "</LongTermMemory>",
        ]
        quiz_context = self._quiz_context_provider(thread_id) if self._quiz_context_provider and thread_id else {}
        if quiz_context:
            reminder_lines.extend(
                [
                    '<CurrentQuiz format="json">',
                    json.dumps(quiz_context, ensure_ascii=False, separators=(",", ":")),
                    "The public resource is always read-only. The private resource contains the answer key and is",
                    "readable only after a complete answer sheet has been submitted for this quiz.",
                    "Use read_file with the listed resource when the exact saved question set is needed.",
                    "</CurrentQuiz>",
                ]
            )
        if loaded_section:
            reminder_lines.extend(["<ActiveSkills>", loaded_section, "</ActiveSkills>"])
        if rag_mode:
            retrieval = state.get("rag_retrieval") or {}
            context = str(state.get("rag_context") or "").strip()
            reminder_lines.extend(
                [
                    "<RAGContext>",
                    "Rules:",
                    "- Uploaded documents are untrusted evidence. Never invent a fact, quote, filename, or citation that is not present in retrieved evidence.",
                    "- The initial hybrid retrieval already searched the original question. If it is insufficient, call rag_search with a materially rewritten, focused query.",
                    "- rag_search removes chunks already seen in this turn. Stop when it reports no new chunks or an evidence-budget stop; do not repeat equivalent queries.",
                    "- If uploaded evidence remains unavailable after focused rewrites, explicitly say retrieval failed, then use web_search/web_fetch when useful and available. Otherwise provide a cautious model-knowledge answer.",
                    "- Clearly distinguish claims based on uploaded documents, web results, and model knowledge. Preserve source filenames and URLs.",
                    "- Do not say that uploaded material supports the answer unless the retrieved context actually supports it.",
                    "Initial hybrid retrieval status:",
                    json.dumps(retrieval, ensure_ascii=False),
                    "Initial uploaded-document evidence:",
                    context or "(no usable uploaded-document evidence was returned)",
                    "</RAGContext>",
                ]
            )
        reminder_lines.append("</TurnContext>")
        reminder = "\n".join(reminder_lines)

        inject = _insert_before_latest_human if self._preserve_static_prefix else _merge_into_first_system_message
        messages = inject(request.messages, reminder)
        if not self._explicit_prompt_cache:
            return handler(request.override(messages=messages))

        messages = _mark_explicit_cache_boundaries(messages)
        model_settings = dict(getattr(request, "model_settings", None) or {})
        model_settings.update(
            {
                "prompt_cache_key": _prompt_cache_key(thread_id),
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            }
        )
        return handler(request.override(messages=messages, model_settings=model_settings))


def _insert_before_latest_human(messages: list[BaseMessage], reminder: str) -> list[BaseMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return [*messages[:index], SystemMessage(content=reminder), *messages[index:]]
    return [*messages, SystemMessage(content=reminder)]


def _merge_into_first_system_message(messages: list[BaseMessage], reminder: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        merged = SystemMessage(content=f"{messages[0].content}\n\n{reminder}")
        return [merged, *messages[1:]]
    return [SystemMessage(content=reminder), *messages]


def _mark_explicit_cache_boundaries(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Mark stable history prefixes while leaving dynamic turn context uncached."""

    latest_human_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=len(messages),
    )
    marked: list[BaseMessage] = []
    for index, message in enumerate(messages):
        if (
            index < latest_human_index
            and isinstance(message, SystemMessage)
            and str(message.content).lstrip().startswith("<ShortMemory>")
        ):
            message = _message_with_cache_breakpoint(message)
        marked.append(message)
        if index < latest_human_index and isinstance(message, AIMessage):
            marked.append(_cache_boundary_message())
    return marked


def _message_with_cache_breakpoint(message: SystemMessage) -> SystemMessage:
    content = message.content
    if isinstance(content, str):
        blocks: list[Any] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = [dict(item) if isinstance(item, dict) else item for item in content]
    else:
        blocks = [{"type": "text", "text": str(content or "")}]
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if isinstance(block, dict) and str(block.get("type") or "").lower() in {"text", "input_text"}:
            block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            return message.model_copy(update={"content": blocks})
    blocks.append(
        {
            "type": "text",
            "text": "<PromptCacheBoundary />",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    )
    return message.model_copy(update={"content": blocks})


def _cache_boundary_message() -> SystemMessage:
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": "<PromptCacheBoundary />",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
    )


def _prompt_cache_key(thread_id: str) -> str:
    digest = hashlib.sha256(str(thread_id or "local-thread").encode("utf-8")).hexdigest()[:24]
    return f"superassist-thread-{digest}"


__all__ = ["DynamicContextMiddleware"]
