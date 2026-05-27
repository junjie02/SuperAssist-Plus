from __future__ import annotations

import re
from collections.abc import Mapping
import os
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from superassist_plus.config import Settings, get_settings

_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


class FallbackChatModel(BaseChatModel):
    """Deterministic local fallback used when no API key is configured."""

    @property
    def _llm_type(self) -> str:
        return "superassist-plus-fallback"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[no-untyped-def]
        """Accept LangChain tool binding while intentionally never calling tools."""

        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        last_user = ""
        for message in reversed(messages):
            if getattr(message, "type", "") == "human":
                last_user = str(message.content)
                break
        content = (
            "SuperAssist-Plus is running in fallback mode because no model API key is configured. "
            f"Latest user request: {last_user}"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class MiniMaxCompatibleChatModel(ChatOpenAI):
    """MiniMax adapter for OpenAI-compatible chat with reasoning split support."""

    @property
    def _llm_type(self) -> str:
        return "superassist-plus-minimax"

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload and "max_tokens" not in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            payload["extra_body"] = {**extra_body, "reasoning_split": True}
        else:
            payload["extra_body"] = {"reasoning_split": True}
        debug_path = os.getenv("SUPERASSIST_PLUS_DEBUG_MINIMAX_PAYLOAD")
        if debug_path:
            with open(debug_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        return payload

    def _create_chat_result(self, response, generation_info=None):  # type: ignore[no-untyped-def]
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices", [])
        generations: list[ChatGeneration] = []
        for index, generation in enumerate(result.generations):
            choice = choices[index] if index < len(choices) else {}
            message = generation.message
            if isinstance(message, AIMessage):
                updated_message = _message_with_minimax_reasoning(message, choice)
                generation = ChatGeneration(message=updated_message, generation_info=generation.generation_info)
            generations.append(generation)
        return ChatResult(generations=generations, llm_output=result.llm_output)


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if not settings.api_key:
        return FallbackChatModel()
    if settings.model_provider.lower() != "openai":
        raise ValueError(f"Unsupported model provider: {settings.model_provider}")
    kwargs = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "timeout": 60,
        "max_retries": 2,
    }
    temperature = settings.temperature
    if temperature is None and "minimax" in settings.model.lower():
        temperature = 1.0
    if temperature is not None:
        kwargs["temperature"] = temperature
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    model_class = MiniMaxCompatibleChatModel if is_minimax_model(settings.model, settings.base_url) else ChatOpenAI
    return model_class(
        **kwargs,
    )


def is_minimax_model(model: str, base_url: str = "") -> bool:
    return "minimax" in model.lower() or "minimax" in base_url.lower()


def _message_with_minimax_reasoning(message: AIMessage, choice: Any) -> AIMessage:
    content = message.content if isinstance(message.content, str) else None
    cleaned_content = content
    inline_reasoning = None
    if isinstance(content, str):
        cleaned_content, inline_reasoning = _strip_inline_think_tags(content)
    choice_message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
    split_reasoning = _extract_reasoning_text(choice_message.get("reasoning_details"))
    reasoning = _merge_reasoning(split_reasoning, inline_reasoning)
    updated = message
    if cleaned_content is not None and cleaned_content != message.content:
        updated = updated.model_copy(update={"content": cleaned_content})
    if reasoning:
        additional_kwargs = dict(updated.additional_kwargs)
        additional_kwargs["reasoning_content"] = _merge_reasoning(
            additional_kwargs.get("reasoning_content"),
            reasoning,
        )
        updated = updated.model_copy(update={"additional_kwargs": additional_kwargs})
    return updated


def _extract_reasoning_text(reasoning_details: Any) -> str | None:
    if not isinstance(reasoning_details, list):
        return None
    parts: list[str] = []
    for item in reasoning_details:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts) if parts else None


def _strip_inline_think_tags(content: str) -> tuple[str, str | None]:
    reasoning_parts: list[str] = []

    def replace(match: re.Match[str]) -> str:
        reasoning = match.group(1).strip()
        if reasoning:
            reasoning_parts.append(reasoning)
        return ""

    cleaned = _THINK_TAG_RE.sub(replace, content).strip()
    return cleaned, "\n\n".join(reasoning_parts) if reasoning_parts else None


def _merge_reasoning(*values: str | None) -> str | None:
    merged: list[str] = []
    for value in values:
        if not value:
            continue
        normalized = value.strip()
        if normalized and normalized not in merged:
            merged.append(normalized)
    return "\n\n".join(merged) if merged else None
