from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import set_key
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from superassist.config import Settings

_ENV_WRITE_LOCK = Lock()


class MemorySettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_writer_enabled: bool
    debounce_seconds: float = Field(ge=0, le=3600)
    top_k: int = Field(ge=1, le=1000)
    candidate_pool_size: int = Field(ge=1, le=10000)
    read_use_ppr: bool
    read_entry_points: int = Field(ge=1, le=1000)
    read_max_depth: int = Field(ge=1, le=20)
    read_bfs_weight: float = Field(ge=0, le=1)
    read_ppr_weight: float = Field(ge=0, le=1)
    read_bfs_decay: float = Field(ge=0, le=1)
    reinforce_similarity: float = Field(ge=0, le=1)
    concept_merge_similarity: float = Field(ge=0, le=1)
    completion_similarity: float = Field(ge=0, le=1)
    completion_top_k: int = Field(ge=1, le=1000)
    decay_lambda: float = Field(ge=0, le=10)
    edge_delete_threshold: float = Field(ge=0, le=1)
    short_token_limit: int = Field(ge=100, le=10_000_000)
    short_keep_recent_turns: int = Field(ge=0, le=10000)
    short_summary_target_tokens: int = Field(ge=1, le=1_000_000)

    @model_validator(mode="after")
    def validate_related_values(self) -> "MemorySettingsPayload":
        if self.candidate_pool_size < self.top_k:
            raise ValueError("candidate_pool_size must be greater than or equal to top_k")
        if self.read_bfs_weight + self.read_ppr_weight <= 0:
            raise ValueError("read_bfs_weight and read_ppr_weight cannot both be zero")
        if self.short_summary_target_tokens >= self.short_token_limit:
            raise ValueError("short_summary_target_tokens must be less than short_token_limit")
        return self


class FeishuSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_id: str = Field(max_length=512)
    app_secret: str | None = Field(default=None, max_length=4096)
    domain: str = Field(min_length=1, max_length=2048)
    allowed_open_ids: str = Field(max_length=10000)
    mention_only: bool
    active_session_seconds: int = Field(ge=10, le=86400)
    activation_debounce_seconds: float = Field(ge=0, le=30)
    activation_max_wait_seconds: float = Field(ge=0.1, le=60)
    max_images_per_activation: int = Field(ge=1, le=100)

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("https://", "http://")):
            raise ValueError("domain must start with http:// or https://")
        return value


class SkillSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_ttl_seconds: int = Field(ge=30, le=86400)


class WeComSettingsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = Field(max_length=512)
    bot_secret: str | None = Field(default=None, max_length=4096)
    allowed_user_ids: str = Field(max_length=10000)
    user_id_map: str = Field(max_length=50000)
    rag_mode_default: bool
    max_concurrent: int = Field(ge=1, le=32)
    stream_interval_ms: int = Field(ge=100, le=5000)
    ai_engine_url: str = Field(min_length=1, max_length=2048)
    rpa_allowed_groups: str = Field(max_length=10000)
    rpa_trigger_prefixes: str = Field(max_length=10000)
    rpa_poll_interval_seconds: float = Field(ge=0.5, le=30)
    rpa_reply_max_chars: int = Field(ge=100, le=10000)

    @field_validator("ai_engine_url")
    @classmethod
    def validate_ai_engine_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("https://", "http://")):
            raise ValueError("ai_engine_url must start with http:// or https://")
        return value

    @field_validator("user_id_map")
    @classmethod
    def validate_user_id_map(cls, value: str) -> str:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("user_id_map must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("user_id_map must be a JSON object")
        if any(
            not isinstance(wecom_id, str)
            or not wecom_id.strip()
            or not isinstance(user_id, str)
            or not user_id.strip()
            for wecom_id, user_id in parsed.items()
        ):
            raise ValueError("user_id_map keys and values must be non-empty strings")
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


class SettingsUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory: MemorySettingsPayload
    skills: SkillSettingsPayload
    feishu: FeishuSettingsPayload
    wecom: WeComSettingsPayload


_MEMORY_FIELDS = {
    "llm_writer_enabled": ("memory_llm_writer_enabled", "SUPERASSIST_MEMORY_LLM_WRITER_ENABLED"),
    "debounce_seconds": ("memory_debounce_seconds", "SUPERASSIST_MEMORY_DEBOUNCE_SECONDS"),
    "top_k": ("memory_top_k", "SUPERASSIST_MEMORY_TOP_K"),
    "candidate_pool_size": ("memory_candidate_pool_size", "SUPERASSIST_MEMORY_CANDIDATE_POOL_SIZE"),
    "read_use_ppr": ("memory_read_use_ppr", "SUPERASSIST_MEMORY_READ_USE_PPR"),
    "read_entry_points": ("memory_read_entry_points", "SUPERASSIST_MEMORY_READ_ENTRY_POINTS"),
    "read_max_depth": ("memory_read_max_depth", "SUPERASSIST_MEMORY_READ_MAX_DEPTH"),
    "read_bfs_weight": ("memory_read_bfs_weight", "SUPERASSIST_MEMORY_READ_BFS_WEIGHT"),
    "read_ppr_weight": ("memory_read_ppr_weight", "SUPERASSIST_MEMORY_READ_PPR_WEIGHT"),
    "read_bfs_decay": ("memory_read_bfs_decay", "SUPERASSIST_MEMORY_READ_BFS_DECAY"),
    "reinforce_similarity": ("memory_reinforce_similarity", "SUPERASSIST_MEMORY_REINFORCE_SIMILARITY"),
    "concept_merge_similarity": (
        "memory_concept_merge_similarity",
        "SUPERASSIST_MEMORY_CONCEPT_MERGE_SIMILARITY",
    ),
    "completion_similarity": ("memory_completion_similarity", "SUPERASSIST_MEMORY_COMPLETION_SIMILARITY"),
    "completion_top_k": ("memory_completion_top_k", "SUPERASSIST_MEMORY_COMPLETION_TOP_K"),
    "decay_lambda": ("memory_decay_lambda", "SUPERASSIST_MEMORY_DECAY_LAMBDA"),
    "edge_delete_threshold": ("memory_edge_delete_threshold", "SUPERASSIST_MEMORY_EDGE_DELETE_THRESHOLD"),
    "short_token_limit": ("short_memory_token_limit", "SUPERASSIST_SHORT_MEMORY_TOKEN_LIMIT"),
    "short_keep_recent_turns": (
        "short_memory_keep_recent_turns",
        "SUPERASSIST_SHORT_MEMORY_KEEP_RECENT_TURNS",
    ),
    "short_summary_target_tokens": (
        "short_memory_summary_target_tokens",
        "SUPERASSIST_SHORT_MEMORY_SUMMARY_TARGET_TOKENS",
    ),
}

_FEISHU_FIELDS = {
    "app_id": ("feishu_app_id", "SUPERASSIST_FEISHU_APP_ID"),
    "domain": ("feishu_domain", "SUPERASSIST_FEISHU_DOMAIN"),
    "allowed_open_ids": ("feishu_allowed_open_ids", "SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS"),
    "mention_only": ("feishu_mention_only", "SUPERASSIST_FEISHU_MENTION_ONLY"),
    "active_session_seconds": (
        "feishu_active_session_seconds",
        "SUPERASSIST_FEISHU_ACTIVE_SESSION_SECONDS",
    ),
    "activation_debounce_seconds": (
        "feishu_activation_debounce_seconds",
        "SUPERASSIST_FEISHU_ACTIVATION_DEBOUNCE_SECONDS",
    ),
    "activation_max_wait_seconds": (
        "feishu_activation_max_wait_seconds",
        "SUPERASSIST_FEISHU_ACTIVATION_MAX_WAIT_SECONDS",
    ),
    "max_images_per_activation": (
        "feishu_max_images_per_activation",
        "SUPERASSIST_FEISHU_MAX_IMAGES_PER_ACTIVATION",
    ),
}

_SKILL_FIELDS = {
    "active_ttl_seconds": ("skill_active_ttl_seconds", "SUPERASSIST_SKILL_ACTIVE_TTL_SECONDS"),
}

_WECOM_FIELDS = {
    "bot_id": ("wecom_bot_id", "SUPERASSIST_WECOM_BOT_ID"),
    "allowed_user_ids": ("wecom_allowed_user_ids", "SUPERASSIST_WECOM_ALLOWED_USER_IDS"),
    "user_id_map": ("wecom_user_id_map", "SUPERASSIST_WECOM_USER_ID_MAP"),
    "rag_mode_default": ("wecom_rag_mode_default", "SUPERASSIST_WECOM_RAG_MODE_DEFAULT"),
    "max_concurrent": ("wecom_max_concurrent", "SUPERASSIST_WECOM_MAX_CONCURRENT"),
    "stream_interval_ms": ("wecom_stream_interval_ms", "SUPERASSIST_WECOM_STREAM_INTERVAL_MS"),
    "ai_engine_url": ("wecom_ai_engine_url", "SUPERASSIST_WECOM_AI_ENGINE_URL"),
    "rpa_allowed_groups": ("wecom_rpa_allowed_groups", "SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS"),
    "rpa_trigger_prefixes": ("wecom_rpa_trigger_prefixes", "SUPERASSIST_WECOM_RPA_TRIGGER_PREFIXES"),
    "rpa_poll_interval_seconds": (
        "wecom_rpa_poll_interval_seconds",
        "SUPERASSIST_WECOM_RPA_POLL_INTERVAL_SECONDS",
    ),
    "rpa_reply_max_chars": ("wecom_rpa_reply_max_chars", "SUPERASSIST_WECOM_RPA_REPLY_MAX_CHARS"),
}


def register_settings_routes(app: FastAPI, settings: Settings, env_path: Path) -> None:
    @app.get("/internal/settings")
    def internal_settings() -> dict[str, Any]:
        return settings_payload(settings)

    @app.put("/internal/settings")
    def update_internal_settings(request: SettingsUpdatePayload) -> dict[str, Any]:
        try:
            return apply_settings_update(settings, request, env_path)
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Failed to persist settings") from exc


def settings_payload(
    settings: Settings,
    *,
    feishu_restart_required: bool = False,
    wecom_restart_required: bool = False,
) -> dict[str, Any]:
    memory = {
        public_name: getattr(settings, attribute_name)
        for public_name, (attribute_name, _env_name) in _MEMORY_FIELDS.items()
    }
    skills = {
        public_name: getattr(settings, attribute_name)
        for public_name, (attribute_name, _env_name) in _SKILL_FIELDS.items()
    }
    feishu = {
        public_name: getattr(settings, attribute_name)
        for public_name, (attribute_name, _env_name) in _FEISHU_FIELDS.items()
    }
    feishu["app_secret_configured"] = bool(settings.feishu_app_secret)
    wecom = {
        public_name: getattr(settings, attribute_name)
        for public_name, (attribute_name, _env_name) in _WECOM_FIELDS.items()
    }
    wecom["bot_secret_configured"] = bool(settings.wecom_bot_secret)
    return {
        "memory": memory,
        "skills": skills,
        "feishu": feishu,
        "wecom": wecom,
        "meta": {
            "memory_applied": True,
            "feishu_restart_required": feishu_restart_required,
            "wecom_restart_required": wecom_restart_required,
        },
    }


def apply_settings_update(
    settings: Settings,
    payload: SettingsUpdatePayload,
    env_path: Path,
) -> dict[str, Any]:
    updates: list[tuple[str, str, Any]] = []
    for public_name, (attribute_name, env_name) in _MEMORY_FIELDS.items():
        updates.append((attribute_name, env_name, getattr(payload.memory, public_name)))
    for public_name, (attribute_name, env_name) in _SKILL_FIELDS.items():
        updates.append((attribute_name, env_name, getattr(payload.skills, public_name)))
    for public_name, (attribute_name, env_name) in _FEISHU_FIELDS.items():
        updates.append((attribute_name, env_name, getattr(payload.feishu, public_name)))
    if payload.feishu.app_secret is not None:
        updates.append(("feishu_app_secret", "SUPERASSIST_FEISHU_APP_SECRET", payload.feishu.app_secret))
    for public_name, (attribute_name, env_name) in _WECOM_FIELDS.items():
        updates.append((attribute_name, env_name, getattr(payload.wecom, public_name)))
    if payload.wecom.bot_secret is not None:
        updates.append(("wecom_bot_secret", "SUPERASSIST_WECOM_BOT_SECRET", payload.wecom.bot_secret))

    feishu_attributes = {attribute_name for attribute_name, _env_name in _FEISHU_FIELDS.values()}
    feishu_attributes.add("feishu_app_secret")
    feishu_changed = any(
        getattr(settings, attribute_name) != value
        for attribute_name, _env_name, value in updates
        if attribute_name in feishu_attributes
    )
    wecom_attributes = {attribute_name for attribute_name, _env_name in _WECOM_FIELDS.values()}
    wecom_attributes.add("wecom_bot_secret")
    wecom_changed = any(
        getattr(settings, attribute_name) != value
        for attribute_name, _env_name, value in updates
        if attribute_name in wecom_attributes
    )

    env_path.parent.mkdir(parents=True, exist_ok=True)
    with _ENV_WRITE_LOCK:
        for _attribute_name, env_name, value in updates:
            set_key(str(env_path), env_name, _serialize_env_value(value), quote_mode="auto")

    for attribute_name, _env_name, value in updates:
        setattr(settings, attribute_name, value)

    return settings_payload(
        settings,
        feishu_restart_required=feishu_changed,
        wecom_restart_required=wecom_changed,
    )


def _serialize_env_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
