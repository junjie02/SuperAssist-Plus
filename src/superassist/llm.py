from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from superassist.config import Settings, get_settings

_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_MODEL_INPUT_LOG_LOCK = threading.Lock()
_MODEL_INPUT_LOG_BACKUPS = 3
logger = logging.getLogger(__name__)


class _FailoverRouteState:
    def __init__(self, selected_index: int = 0) -> None:
        self.selected_index = selected_index


class OneSecondRetryChatModel(ChatOpenAI):
    """OpenAI-compatible chat model with one explicit 1s retry around generation."""

    _model_input_log_path: Path | None = PrivateAttr(default=None)
    _model_input_log_max_bytes: int = PrivateAttr(default=50 * 1024 * 1024)
    _model_input_log_call_kind: str = PrivateAttr(default="model")

    def configure_input_logging(self, *, path: Path | None, max_bytes: int, call_kind: str = "model") -> None:
        self._model_input_log_path = path
        self._model_input_log_max_bytes = max(1024, max_bytes)
        self._model_input_log_call_kind = call_kind

    def _prepare_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        payload = self._prepare_request_payload(input_, stop=stop, **kwargs)
        self._log_model_input(payload)
        return payload

    def _log_model_input(self, payload: dict[str, Any]) -> None:
        if self._model_input_log_path is None:
            return
        try:
            _append_model_input_log(
                self._model_input_log_path,
                payload,
                max_bytes=self._model_input_log_max_bytes,
                call_kind=self._model_input_log_call_kind,
            )
        except Exception:
            # Observability must never block or alter a model request.
            pass

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            if not _should_retry_same_route(exc):
                raise
            time.sleep(1)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _convert_chunk_to_generation_chunk(  # type: ignore[no-untyped-def]
        self, chunk, default_chunk_class, base_generation_info
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return generation
        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        delta = choices[0].get("delta") if choices else None
        if not isinstance(delta, Mapping):
            return generation
        reasoning = _merge_reasoning(
            delta.get("reasoning_content") if isinstance(delta.get("reasoning_content"), str) else None,
            delta.get("reasoning") if isinstance(delta.get("reasoning"), str) else None,
            _extract_reasoning_text(delta.get("reasoning_details")),
        )
        if not reasoning:
            return generation
        additional_kwargs = dict(generation.message.additional_kwargs)
        additional_kwargs["reasoning_content"] = reasoning
        message = generation.message.model_copy(update={"additional_kwargs": additional_kwargs})
        return generation.model_copy(update={"message": message})


class FailoverChatModel(BaseChatModel):
    """Sticky ordered failover across OpenAI-compatible chat routes."""

    _routes: list[Runnable] = PrivateAttr(default_factory=list)
    _route_names: list[str] = PrivateAttr(default_factory=list)
    _route_state: _FailoverRouteState = PrivateAttr(default_factory=_FailoverRouteState)

    def __init__(self, routes: list[Runnable], route_names: list[str]) -> None:
        if not routes or len(routes) != len(route_names):
            raise ValueError("FailoverChatModel requires equally sized non-empty routes and names")
        super().__init__()
        self._routes = routes
        self._route_names = route_names

    @property
    def _llm_type(self) -> str:
        return "superassist-failover"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"routes": list(self._route_names)}

    @property
    def active_route(self) -> str:
        return self._route_names[self._route_state.selected_index]

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[no-untyped-def]
        bound_routes = [
            route.bind_tools(tools, tool_choice=tool_choice, **kwargs)
            for route in self._routes
        ]
        bound = FailoverChatModel(bound_routes, list(self._route_names))
        # create_agent may bind tools again for each model node. Sharing this
        # tiny state keeps a selected fallback sticky through the tool loop.
        bound._route_state = self._route_state
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Exception | None = None
        for index in range(self._route_state.selected_index, len(self._routes)):
            route = self._routes[index]
            route_messages = _route_messages(messages, self._route_names[index])
            try:
                message = route.invoke(route_messages, stop=stop, **_route_kwargs(kwargs, fallback=index > 0))
            except Exception as exc:  # noqa: BLE001 - route boundary classifies and records provider failures
                last_error = exc
                if not _should_failover(exc) or index == len(self._routes) - 1:
                    raise
                logger.warning(
                    "Model route failed; switching from %s to %s error_type=%s",
                    self._route_names[index],
                    self._route_names[index + 1],
                    type(exc).__name__,
                )
                self._route_state.selected_index = index + 1
                continue
            self._route_state.selected_index = index
            if not isinstance(message, AIMessage):
                raise TypeError(f"Model route {self._route_names[index]} returned {type(message).__name__}")
            return ChatResult(
                generations=[ChatGeneration(message=_with_model_route(message, self._route_names[index]))]
            )
        assert last_error is not None
        raise last_error

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ):
        last_error: Exception | None = None
        for index in range(self._route_state.selected_index, len(self._routes)):
            emitted = False
            route_messages = _route_messages(messages, self._route_names[index])
            try:
                for chunk in self._routes[index].stream(
                    route_messages,
                    stop=stop,
                    **_route_kwargs(kwargs, fallback=index > 0),
                ):
                    emitted = True
                    if isinstance(chunk, AIMessageChunk):
                        chunk = _with_model_route(chunk, self._route_names[index])
                    yield ChatGenerationChunk(message=chunk)
                self._route_state.selected_index = index
                return
            except Exception as exc:  # noqa: BLE001 - see synchronous route boundary above
                last_error = exc
                if emitted or not _should_failover(exc) or index == len(self._routes) - 1:
                    raise
                logger.warning(
                    "Streaming model route failed before first chunk; switching from %s to %s error_type=%s",
                    self._route_names[index],
                    self._route_names[index + 1],
                    type(exc).__name__,
                )
                self._route_state.selected_index = index + 1
        assert last_error is not None
        raise last_error


class FallbackChatModel(BaseChatModel):
    """Deterministic local fallback used when no API key is configured."""

    @property
    def _llm_type(self) -> str:
        return "superassist-fallback"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs) -> Runnable:  # type: ignore[no-untyped-def]
        """Accept LangChain tool binding while intentionally never calling tools."""

        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        if messages and "compress conversation history" in str(messages[0].content).lower():
            content = self._fallback_summary(messages)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        last_user = ""
        for message in reversed(messages):
            if getattr(message, "type", "") == "human":
                last_user = str(message.content)
                break
        content = (
            "SuperAssist is running in fallback mode because no model API key is configured. "
            f"Latest user request: {last_user}"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @staticmethod
    def _fallback_summary(messages) -> str:  # type: ignore[no-untyped-def]
        last_human = ""
        for message in reversed(messages):
            if getattr(message, "type", "") == "human":
                last_human = str(message.content)
                break
        lines = [line.strip() for line in last_human.splitlines() if line.strip()]
        useful = [
            line
            for line in lines
            if line.startswith(("user:", "assistant:", "Tool event:", "Loaded skills:", "Previous summary:"))
        ][:20]
        if not useful:
            useful = ["Conversation history was compressed in fallback mode."]
        return "## Conversation Summary\n" + "\n".join(f"- {line}" for line in useful)


class MiniMaxCompatibleChatModel(OneSecondRetryChatModel):
    """MiniMax adapter for OpenAI-compatible chat with reasoning split support."""

    @property
    def _llm_type(self) -> str:
        return "superassist-minimax"

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[no-untyped-def]
        payload = self._prepare_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload and "max_tokens" not in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        _strip_message_names(payload)
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            payload["extra_body"] = {**extra_body, "reasoning_split": True}
        else:
            payload["extra_body"] = {"reasoning_split": True}
        debug_path = os.getenv("SUPERASSIST_DEBUG_MINIMAX_PAYLOAD")
        if debug_path:
            with open(debug_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        self._log_model_input(payload)
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
        routes: list[Runnable] = []
        route_names: list[str] = []
        if settings.claude_fallback_api_key and settings.claude_fallback_base_url:
            routes.append(
                _create_compatible_fallback_model(
                    model=settings.claude_fallback_model,
                    api_key=settings.claude_fallback_api_key,
                    base_url=settings.claude_fallback_base_url,
                    settings=settings,
                    call_kind="lead_agent_claude_fallback",
                )
            )
            route_names.append(f"claude:{settings.claude_fallback_model}")
        if settings.deepseek_fallback_api_key and settings.deepseek_fallback_base_url:
            routes.append(
                _create_compatible_fallback_model(
                    model=settings.deepseek_fallback_model,
                    api_key=settings.deepseek_fallback_api_key,
                    base_url=settings.deepseek_fallback_base_url,
                    settings=settings,
                    call_kind="lead_agent_deepseek_fallback",
                )
            )
            route_names.append(f"deepseek:{settings.deepseek_fallback_model}")
        if not routes:
            return FallbackChatModel()
        return routes[0] if len(routes) == 1 else FailoverChatModel(routes, route_names)
    if settings.model_provider.lower() != "openai":
        raise ValueError(f"Unsupported model provider: {settings.model_provider}")
    kwargs = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "timeout": 60,
        "max_retries": 2,
        "stream_usage": True,
        "use_responses_api": settings.use_responses_api,
    }
    if settings.use_responses_api:
        # Preserve the full multimodal turn across tool-loop requests.
        kwargs["use_previous_response_id"] = False
    if is_gpt_5_6_model(settings.model) and settings.use_responses_api:
        reasoning: dict[str, str] = {"effort": settings.reasoning_effort}
        if settings.reasoning_effort != "none":
            reasoning["summary"] = "detailed"
        kwargs.update(
            {
                "reasoning": reasoning,
            }
        )
    elif is_gpt_5_6_model(settings.model):
        kwargs.update(
            {
                "use_responses_api": False,
                "reasoning_effort": settings.reasoning_effort,
            }
        )
    temperature = settings.temperature
    if temperature is None and "minimax" in settings.model.lower():
        temperature = 1.0
    if temperature is not None:
        kwargs["temperature"] = temperature
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    model_class = MiniMaxCompatibleChatModel if is_minimax_model(settings.model, settings.base_url) else OneSecondRetryChatModel
    model = model_class(**kwargs)
    model.configure_input_logging(
        path=settings.model_input_log_path if settings.model_input_log_enabled else None,
        max_bytes=settings.model_input_log_max_bytes,
        call_kind="lead_agent",
    )
    routes: list[Runnable] = [model]
    route_names = [f"primary:{settings.model}"]
    if settings.claude_fallback_api_key and settings.claude_fallback_base_url:
        routes.append(
            _create_compatible_fallback_model(
                model=settings.claude_fallback_model,
                api_key=settings.claude_fallback_api_key,
                base_url=settings.claude_fallback_base_url,
                settings=settings,
                call_kind="lead_agent_claude_fallback",
            )
        )
        route_names.append(f"claude:{settings.claude_fallback_model}")
    deepseek_api_key = settings.deepseek_fallback_api_key
    deepseek_base_url = settings.deepseek_fallback_base_url
    if deepseek_api_key and deepseek_base_url:
        routes.append(
            _create_compatible_fallback_model(
                model=settings.deepseek_fallback_model,
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                settings=settings,
                call_kind="lead_agent_deepseek_fallback",
            )
        )
        route_names.append(f"deepseek:{settings.deepseek_fallback_model}")
    return model if len(routes) == 1 else FailoverChatModel(routes, route_names)


def _create_compatible_fallback_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
    settings: Settings,
    call_kind: str,
) -> OneSecondRetryChatModel:
    fallback = OneSecondRetryChatModel(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=1,
        stream_usage=True,
        use_responses_api=settings.use_responses_api,
        use_previous_response_id=False,
    )
    fallback.configure_input_logging(
        path=settings.model_input_log_path if settings.model_input_log_enabled else None,
        max_bytes=settings.model_input_log_max_bytes,
        call_kind=call_kind,
    )
    return fallback


def create_memory_model(
    settings: Settings | None = None,
    *,
    call_kind: str = "memory_updater",
) -> BaseChatModel:
    """Create the independent OpenAI-compatible model used for memory work."""

    settings = settings or get_settings()
    api_key = settings.memory_api_key or settings.api_key
    if not api_key:
        return FallbackChatModel()
    kwargs: dict[str, Any] = {
        "model": settings.memory_model,
        "api_key": api_key,
        "base_url": settings.memory_base_url or settings.base_url,
        "timeout": 60,
        "max_retries": 2,
        "stream_usage": True,
        "use_responses_api": settings.use_responses_api,
        "use_previous_response_id": False,
    }
    if settings.memory_max_tokens is not None:
        kwargs["max_tokens"] = settings.memory_max_tokens
    model = OneSecondRetryChatModel(**kwargs)
    model.configure_input_logging(
        path=settings.model_input_log_path if settings.model_input_log_enabled else None,
        max_bytes=settings.model_input_log_max_bytes,
        call_kind=call_kind,
    )
    return model


def is_minimax_model(model: str, base_url: str = "") -> bool:
    return "minimax" in model.lower() or "minimax" in base_url.lower()


def is_gpt_5_6_model(model: str) -> bool:
    return model.lower().startswith("gpt-5.6")


def _should_failover(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {400, 413, 422}:
            return False
        return status_code in {401, 403, 404, 405, 408, 409, 429} or status_code >= 500
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    non_retryable_markers = (
        "context length",
        "maximum context",
        "content policy",
        "invalid tool",
        "invalid_request",
    )
    if any(marker in message for marker in non_retryable_markers):
        return False
    return any(
        marker in name or marker in message
        for marker in ("timeout", "connection", "ratelimit", "temporar", "unavailable", "overload")
    ) or isinstance(exc, RuntimeError)


def _should_retry_same_route(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500
    return _should_failover(exc)


def _with_model_route(message: Any, route_name: str) -> Any:
    metadata = dict(getattr(message, "response_metadata", {}) or {})
    metadata["superassist_model_route"] = route_name
    return message.model_copy(update={"response_metadata": metadata})


def _route_kwargs(kwargs: dict[str, Any], *, fallback: bool) -> dict[str, Any]:
    if not fallback:
        return kwargs
    unsupported = {"prompt_cache_key", "prompt_cache_options"}
    return {key: value for key, value in kwargs.items() if key not in unsupported}


def _route_messages(messages: list[BaseMessage], route_name: str) -> list[BaseMessage]:
    if not route_name.startswith("deepseek:"):
        return messages
    rendered: list[BaseMessage] = []
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            rendered.append(message)
            continue
        kept: list[Any] = []
        removed_images = 0
        for item in content:
            item_type = str(item.get("type") or "").lower() if isinstance(item, dict) else ""
            if item_type in {"image", "image_url", "input_image"}:
                removed_images += 1
                continue
            kept.append(item)
        if removed_images:
            kept.append(
                {
                    "type": "text",
                    "text": (
                        f"<VisionDegraded omitted_images=\"{removed_images}\">"
                        "This fallback model did not receive original image pixels. Use only OCR and existing "
                        "visual descriptions, and disclose this limitation in the answer."
                        "</VisionDegraded>"
                    ),
                }
            )
        rendered.append(message.model_copy(update={"content": kept}))
    return rendered


def _strip_message_names(payload: dict[str, Any]) -> None:
    """MiniMax's OpenAI-compatible endpoint rejects named chat messages."""

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict):
            message.pop("name", None)


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


def _append_model_input_log(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int,
    call_kind: str = "model",
) -> None:
    sanitized_payload = _sanitize_log_value(payload)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": f"model_input_{uuid4().hex}",
        "call_kind": call_kind,
        "estimated_input_tokens": _estimate_payload_tokens(sanitized_payload),
        "input_manifest": _build_input_manifest(sanitized_payload),
        "payload": sanitized_payload,
    }
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    encoded_size = len(line.encode("utf-8"))

    with _MODEL_INPUT_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size + encoded_size > max_bytes:
            _rotate_model_input_log(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _rotate_model_input_log(path: Path) -> None:
    for index in range(_MODEL_INPUT_LOG_BACKUPS, 0, -1):
        source = path if index == 1 else Path(f"{path}.{index - 1}")
        target = Path(f"{path}.{index}")
        if not source.exists():
            continue
        if target.exists():
            target.unlink()
        source.replace(target)


def _sanitize_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_log_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/") and len(value) > 512:
        return {
            "data_url_omitted": True,
            "characters": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    return value


def _estimate_payload_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return max(1, len(tiktoken.get_encoding("o200k_base").encode(text)))


def _build_input_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("input") or payload.get("messages") or []
    roles: dict[str, int] = {}
    types: dict[str, int] = {}
    sections: dict[str, int] = {}
    component_tokens: dict[str, int] = {}
    message_rows: list[tuple[str, str, str]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "none")
            item_type = str(item.get("type") or "message")
            roles[role] = roles.get(role, 0) + 1
            types[item_type] = types.get(item_type, 0) + 1
            content = item.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
            message_rows.append((role, item_type, text))
            for tag, name in (
                ("ShortMemory", "short_memory"),
                ("Summary", "short_memory_summary"),
                ("TurnContext", "turn_context"),
                ("RuntimeContext", "runtime_context"),
                ("LongTermMemory", "long_term_memory"),
                ("ActiveSkills", "active_skills"),
                ("RAGContext", "rag_context"),
                ("MemoryWriteInput", "memory_write_input"),
            ):
                tagged_tokens = _tagged_section_tokens(text, tag)
                if tagged_tokens:
                    sections[name] = sections.get(name, 0) + tagged_tokens
    tools = payload.get("tools")
    if message_rows:
        current_user_index = max(
            (index for index, row in enumerate(message_rows) if row[0] == "user"),
            default=-1,
        )
        for index, (role, item_type, text) in enumerate(message_rows):
            tokens = _estimate_payload_tokens(text)
            if index == 0 and role == "system":
                name = "static_system"
            elif "<ShortMemory>" in text:
                name = "short_memory"
            elif "<TurnContext>" in text:
                name = "turn_context"
            elif "<MemoryWriteInput" in text:
                name = "memory_write_input"
            elif index == current_user_index and role == "user":
                name = "current_user"
            elif item_type != "message":
                name = item_type
            elif role == "user":
                name = "history_user"
            elif role == "assistant":
                name = "history_assistant"
            else:
                name = f"other_{role}"
            component_tokens[name] = component_tokens.get(name, 0) + tokens
    if isinstance(tools, list) and tools:
        component_tokens["tool_schemas"] = _estimate_payload_tokens(tools)
    return {
        "input_items": len(items) if isinstance(items, list) else 0,
        "roles": roles,
        "types": types,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "sections": sections,
        "component_tokens": component_tokens,
        "prompt_cache": {
            "key": str(payload.get("prompt_cache_key") or ""),
            "mode": str((payload.get("prompt_cache_options") or {}).get("mode") or ""),
            "ttl": str((payload.get("prompt_cache_options") or {}).get("ttl") or ""),
            "breakpoints": _count_prompt_cache_breakpoints(items),
        },
    }


def _count_prompt_cache_breakpoints(value: Any) -> int:
    if isinstance(value, dict):
        own = 1 if "prompt_cache_breakpoint" in value else 0
        return own + sum(_count_prompt_cache_breakpoints(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_prompt_cache_breakpoints(item) for item in value)
    return 0


def _tagged_section_tokens(text: str, tag: str) -> int:
    pattern = rf"<{re.escape(tag)}(?:\s[^>]*)?>.*?</{re.escape(tag)}>"
    return sum(_estimate_payload_tokens(match.group(0)) for match in re.finditer(pattern, text, flags=re.DOTALL))
