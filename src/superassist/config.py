from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class Settings(BaseSettings):
    """Runtime settings for SuperAssist."""

    model_provider: str = Field(default="openai", alias="SUPERASSIST_MODEL_PROVIDER")
    model: str = Field(default="gpt-4o-mini", alias="SUPERASSIST_MODEL")
    api_key: str = Field(default="", alias="SUPERASSIST_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="SUPERASSIST_BASE_URL")
    temperature: float | None = Field(default=None, alias="SUPERASSIST_TEMPERATURE")
    max_tokens: int | None = Field(default=None, alias="SUPERASSIST_MAX_TOKENS")
    reasoning_effort: ReasoningEffort = Field(default="medium", alias="SUPERASSIST_REASONING_EFFORT")
    use_responses_api: bool = Field(default=True, alias="SUPERASSIST_USE_RESPONSES_API")
    claude_fallback_model: str = Field(
        default="claude-opus-5",
        alias="SUPERASSIST_CLAUDE_FALLBACK_MODEL",
    )
    claude_fallback_api_key: str = Field(default="", alias="SUPERASSIST_CLAUDE_FALLBACK_API_KEY")
    claude_fallback_base_url: str = Field(
        default="https://code.mmkg.cloud/v1",
        alias="SUPERASSIST_CLAUDE_FALLBACK_BASE_URL",
    )
    deepseek_fallback_model: str = Field(
        default="deepseek-v4-flash",
        alias="SUPERASSIST_DEEPSEEK_FALLBACK_MODEL",
    )
    deepseek_fallback_api_key: str = Field(default="", alias="SUPERASSIST_DEEPSEEK_FALLBACK_API_KEY")
    deepseek_fallback_base_url: str = Field(default="", alias="SUPERASSIST_DEEPSEEK_FALLBACK_BASE_URL")
    prompt_cache_explicit_enabled: bool = Field(
        default=True,
        alias="SUPERASSIST_PROMPT_CACHE_EXPLICIT_ENABLED",
    )
    model_input_log_enabled: bool = Field(default=False, alias="SUPERASSIST_MODEL_INPUT_LOG_ENABLED")
    model_input_log_max_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1024,
        alias="SUPERASSIST_MODEL_INPUT_LOG_MAX_BYTES",
    )
    data_dir: Path = Field(default=Path(".superassist"), alias="SUPERASSIST_DATA_DIR")
    db_url: str = Field(default="", alias="SUPERASSIST_DB_URL")
    redis_enabled: bool = Field(default=False, alias="SUPERASSIST_REDIS_ENABLED")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", alias="SUPERASSIST_REDIS_URL")
    redis_prefix: str = Field(default="superassist", alias="SUPERASSIST_REDIS_PREFIX")
    redis_required: bool = Field(default=False, alias="SUPERASSIST_REDIS_REQUIRED")
    redis_socket_timeout_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        alias="SUPERASSIST_REDIS_SOCKET_TIMEOUT_SECONDS",
    )
    redis_task_ttl_seconds: int = Field(
        default=86400,
        ge=60,
        alias="SUPERASSIST_REDIS_TASK_TTL_SECONDS",
    )
    redis_short_memory_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        alias="SUPERASSIST_REDIS_SHORT_MEMORY_TTL_SECONDS",
    )
    redis_recall_ttl_seconds: int = Field(
        default=120,
        ge=1,
        alias="SUPERASSIST_REDIS_RECALL_TTL_SECONDS",
    )
    api_rate_limit_per_minute: int = Field(
        default=0,
        ge=0,
        alias="SUPERASSIST_API_RATE_LIMIT_PER_MINUTE",
    )
    tool_workspace_dir: Path | None = Field(default=None, alias="SUPERASSIST_TOOL_WORKSPACE_DIR")
    tool_network_enabled: bool = Field(default=True, alias="SUPERASSIST_TOOL_NETWORK_ENABLED")
    tool_shell_enabled: bool = Field(default=False, alias="SUPERASSIST_TOOL_SHELL_ENABLED")
    tool_shell_timeout_seconds: int = Field(default=120, alias="SUPERASSIST_TOOL_SHELL_TIMEOUT_SECONDS")
    tool_shell_output_max_chars: int = Field(default=20000, alias="SUPERASSIST_TOOL_SHELL_OUTPUT_MAX_CHARS")
    max_tool_calls: int = Field(default=8, alias="SUPERASSIST_MAX_TOOL_CALLS")
    enable_tools: bool = Field(default=False, alias="SUPERASSIST_ENABLE_TOOLS")
    image_generation_model: str = Field(default="gpt-image-2", alias="SUPERASSIST_IMAGE_GENERATION_MODEL")
    image_generation_api_key: str = Field(default="", alias="SUPERASSIST_IMAGE_GENERATION_API_KEY")
    image_generation_base_url: str = Field(default="", alias="SUPERASSIST_IMAGE_GENERATION_BASE_URL")
    subagents_enabled: bool = Field(default=True, alias="SUPERASSIST_SUBAGENTS_ENABLED")
    agent_teams_enabled: bool = Field(default=True, alias="SUPERASSIST_AGENT_TEAMS_ENABLED")
    subagent_max_concurrent: int = Field(default=3, alias="SUPERASSIST_SUBAGENT_MAX_CONCURRENT")
    subagent_timeout_seconds: int = Field(default=900, alias="SUPERASSIST_SUBAGENT_TIMEOUT_SECONDS")
    subagent_max_turns: int = Field(default=20, alias="SUPERASSIST_SUBAGENT_MAX_TURNS")
    agents_dir: Path = Field(default=Path("config/agents"), alias="SUPERASSIST_AGENTS_DIR")
    memory_llm_writer_enabled: bool = Field(default=True, alias="SUPERASSIST_MEMORY_LLM_WRITER_ENABLED")
    memory_model: str = Field(default="deepseek-v4-flash", alias="SUPERASSIST_MEMORY_MODEL")
    memory_api_key: str = Field(default="", alias="SUPERASSIST_MEMORY_API_KEY")
    memory_base_url: str = Field(default="", alias="SUPERASSIST_MEMORY_BASE_URL")
    memory_max_tokens: int | None = Field(default=None, alias="SUPERASSIST_MEMORY_MAX_TOKENS")
    short_memory_token_limit: int = Field(default=80000, alias="SUPERASSIST_SHORT_MEMORY_TOKEN_LIMIT")
    short_memory_keep_recent_turns: int = Field(default=30, alias="SUPERASSIST_SHORT_MEMORY_KEEP_RECENT_TURNS")
    short_memory_summary_target_tokens: int = Field(
        default=6000,
        alias="SUPERASSIST_SHORT_MEMORY_SUMMARY_TARGET_TOKENS",
    )
    skill_active_ttl_seconds: int = Field(
        default=300,
        ge=30,
        le=86400,
        alias="SUPERASSIST_SKILL_ACTIVE_TTL_SECONDS",
    )

    memory_reinforce_similarity: float = Field(default=0.85, alias="SUPERASSIST_MEMORY_REINFORCE_SIMILARITY")
    memory_concept_merge_similarity: float = Field(default=0.85, alias="SUPERASSIST_MEMORY_CONCEPT_MERGE_SIMILARITY")
    memory_completion_similarity: float = Field(default=0.30, alias="SUPERASSIST_MEMORY_COMPLETION_SIMILARITY")
    memory_completion_top_k: int = Field(default=5, alias="SUPERASSIST_MEMORY_COMPLETION_TOP_K")
    memory_debounce_seconds: float = Field(default=30.0, alias="SUPERASSIST_MEMORY_DEBOUNCE_SECONDS")
    memory_decay_lambda: float = Field(default=0.005, alias="SUPERASSIST_MEMORY_DECAY_LAMBDA")
    memory_edge_delete_threshold: float = Field(default=0.15, alias="SUPERASSIST_MEMORY_EDGE_DELETE_THRESHOLD")
    memory_top_k: int = Field(default=12, alias="SUPERASSIST_MEMORY_TOP_K")
    memory_candidate_pool_size: int = Field(default=150, alias="SUPERASSIST_MEMORY_CANDIDATE_POOL_SIZE")
    memory_read_use_ppr: bool = Field(default=True, alias="SUPERASSIST_MEMORY_READ_USE_PPR")
    memory_read_entry_points: int = Field(default=10, alias="SUPERASSIST_MEMORY_READ_ENTRY_POINTS")
    memory_read_max_depth: int = Field(default=3, alias="SUPERASSIST_MEMORY_READ_MAX_DEPTH")
    memory_read_bfs_weight: float = Field(default=0.6, alias="SUPERASSIST_MEMORY_READ_BFS_WEIGHT")
    memory_read_ppr_weight: float = Field(default=0.4, alias="SUPERASSIST_MEMORY_READ_PPR_WEIGHT")
    memory_read_bfs_decay: float = Field(default=0.7, alias="SUPERASSIST_MEMORY_READ_BFS_DECAY")
    embedding_provider: str = Field(default="bge", alias="SUPERASSIST_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-base-zh-v1.5", alias="SUPERASSIST_EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="SUPERASSIST_EMBEDDING_DEVICE")
    rag_max_file_size_mb: int = Field(default=25, alias="SUPERASSIST_RAG_MAX_FILE_SIZE_MB")
    rag_max_files_per_batch: int = Field(default=20, alias="SUPERASSIST_RAG_MAX_FILES_PER_BATCH")
    rag_max_attempts: int = Field(default=3, alias="SUPERASSIST_RAG_MAX_ATTEMPTS")
    rag_top_k: int = Field(default=20, alias="SUPERASSIST_RAG_TOP_K")
    rag_chunk_top_k: int = Field(default=10, alias="SUPERASSIST_RAG_CHUNK_TOP_K")
    rag_context_max_chars: int = Field(default=24000, alias="SUPERASSIST_RAG_CONTEXT_MAX_CHARS")
    feishu_app_id: str = Field(default="", alias="SUPERASSIST_FEISHU_APP_ID")
    feishu_app_secret: str = Field(default="", alias="SUPERASSIST_FEISHU_APP_SECRET")
    feishu_domain: str = Field(default="https://open.feishu.cn", alias="SUPERASSIST_FEISHU_DOMAIN")
    feishu_allowed_open_ids: str = Field(default="", alias="SUPERASSIST_FEISHU_ALLOWED_OPEN_IDS")
    feishu_mention_only: bool = Field(default=True, alias="SUPERASSIST_FEISHU_MENTION_ONLY")
    feishu_active_session_seconds: int = Field(
        default=180,
        alias="SUPERASSIST_FEISHU_ACTIVE_SESSION_SECONDS",
    )
    feishu_activation_debounce_seconds: float = Field(
        default=1.5,
        ge=0,
        le=30,
        alias="SUPERASSIST_FEISHU_ACTIVATION_DEBOUNCE_SECONDS",
    )
    feishu_activation_max_wait_seconds: float = Field(
        default=6.0,
        ge=0.1,
        le=60,
        alias="SUPERASSIST_FEISHU_ACTIVATION_MAX_WAIT_SECONDS",
    )
    feishu_max_images_per_activation: int = Field(
        default=12,
        ge=1,
        le=100,
        alias="SUPERASSIST_FEISHU_MAX_IMAGES_PER_ACTIVATION",
    )
    feishu_image_ocr_enabled: bool = Field(
        default=True,
        alias="SUPERASSIST_FEISHU_IMAGE_OCR_ENABLED",
    )
    feishu_image_ocr_max_chars: int = Field(
        default=12000,
        ge=0,
        le=100000,
        alias="SUPERASSIST_FEISHU_IMAGE_OCR_MAX_CHARS",
    )
    feishu_image_context_ttl_seconds: int = Field(
        default=180,
        ge=1,
        alias="SUPERASSIST_FEISHU_IMAGE_CONTEXT_TTL_SECONDS",
    )
    feishu_doc_url_base: str = Field(
        default="https://feishu.cn/docx",
        alias="SUPERASSIST_FEISHU_DOC_URL_BASE",
    )
    daily_brief_enabled: bool = Field(default=False, alias="SUPERASSIST_DAILY_BRIEF_ENABLED")
    daily_brief_times: str = Field(default="07:45,19:45", alias="SUPERASSIST_DAILY_BRIEF_TIMES")
    daily_brief_timezone: str = Field(default="Asia/Shanghai", alias="SUPERASSIST_DAILY_BRIEF_TIMEZONE")
    daily_brief_feishu_chat_ids: str = Field(
        default="",
        alias="SUPERASSIST_DAILY_BRIEF_FEISHU_CHAT_IDS",
    )
    daily_brief_source_file: Path = Field(
        default=Path("config/official_media.toml"),
        alias="SUPERASSIST_DAILY_BRIEF_SOURCE_FILE",
    )
    daily_brief_prompt_file: Path = Field(
        default=Path("prompts/shenlun_daily_brief.md"),
        alias="SUPERASSIST_DAILY_BRIEF_PROMPT_FILE",
    )
    daily_brief_lookback_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        alias="SUPERASSIST_DAILY_BRIEF_LOOKBACK_HOURS",
    )
    daily_brief_catch_up_minutes: int = Field(
        default=10,
        ge=0,
        le=180,
        alias="SUPERASSIST_DAILY_BRIEF_CATCH_UP_MINUTES",
    )
    daily_brief_max_candidates: int = Field(
        default=80,
        ge=10,
        le=300,
        alias="SUPERASSIST_DAILY_BRIEF_MAX_CANDIDATES",
    )
    daily_brief_min_sources: int = Field(
        default=3,
        ge=1,
        le=20,
        alias="SUPERASSIST_DAILY_BRIEF_MIN_SOURCES",
    )
    daily_brief_model: str = Field(
        default="deepseek-v4-flash",
        alias="SUPERASSIST_DAILY_BRIEF_MODEL",
    )
    daily_brief_api_key: str = Field(default="", alias="SUPERASSIST_DAILY_BRIEF_API_KEY")
    daily_brief_base_url: str = Field(default="", alias="SUPERASSIST_DAILY_BRIEF_BASE_URL")
    daily_quiz_enabled: bool = Field(default=True, alias="SUPERASSIST_DAILY_QUIZ_ENABLED")
    daily_quiz_time: str = Field(default="17:00", alias="SUPERASSIST_DAILY_QUIZ_TIME")
    daily_quiz_question_count: int = Field(
        default=10,
        ge=1,
        le=30,
        alias="SUPERASSIST_DAILY_QUIZ_QUESTION_COUNT",
    )
    daily_quiz_notebook_days: int = Field(
        default=3,
        ge=1,
        le=14,
        alias="SUPERASSIST_DAILY_QUIZ_NOTEBOOK_DAYS",
    )
    wecom_bot_id: str = Field(default="", alias="SUPERASSIST_WECOM_BOT_ID")
    wecom_bot_secret: str = Field(default="", alias="SUPERASSIST_WECOM_BOT_SECRET")
    wecom_allowed_user_ids: str = Field(default="", alias="SUPERASSIST_WECOM_ALLOWED_USER_IDS")
    wecom_user_id_map: str = Field(default="{}", alias="SUPERASSIST_WECOM_USER_ID_MAP")
    wecom_rag_mode_default: bool = Field(default=False, alias="SUPERASSIST_WECOM_RAG_MODE_DEFAULT")
    wecom_max_concurrent: int = Field(default=3, alias="SUPERASSIST_WECOM_MAX_CONCURRENT")
    wecom_stream_interval_ms: int = Field(default=300, alias="SUPERASSIST_WECOM_STREAM_INTERVAL_MS")
    wecom_ai_engine_url: str = Field(
        default="http://127.0.0.1:8765",
        alias="SUPERASSIST_WECOM_AI_ENGINE_URL",
    )
    wecom_rpa_allowed_groups: str = Field(default="", alias="SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS")
    wecom_rpa_trigger_prefixes: str = Field(default="", alias="SUPERASSIST_WECOM_RPA_TRIGGER_PREFIXES")
    wecom_rpa_poll_interval_seconds: float = Field(
        default=1.5,
        alias="SUPERASSIST_WECOM_RPA_POLL_INTERVAL_SECONDS",
    )
    wecom_rpa_reply_max_chars: int = Field(default=3000, alias="SUPERASSIST_WECOM_RPA_REPLY_MAX_CHARS")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "superassist.sqlite3"

    @property
    def resolved_tool_workspace_dir(self) -> Path:
        return self.tool_workspace_dir or self.data_dir / "workspace"

    @property
    def resolved_image_generation_api_key(self) -> str:
        return self.image_generation_api_key or self.api_key

    @property
    def resolved_image_generation_base_url(self) -> str:
        return self.image_generation_base_url or self.base_url

    @property
    def generated_image_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "generated-images"

    @property
    def huggingface_cache_dir(self) -> Path:
        return self.data_dir / "huggingface"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def rag_dir(self) -> Path:
        return self.data_dir / "rag"

    @property
    def model_input_log_path(self) -> Path:
        return self.data_dir / "logs" / "model-input.jsonl"

    @property
    def resolved_agents_dir(self) -> Path:
        path = self.agents_dir
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def feishu_allowed_open_id_set(self) -> set[str]:
        return {item.strip() for item in self.feishu_allowed_open_ids.split(",") if item.strip()}

    @property
    def feishu_thread_store_path(self) -> Path:
        return self.data_dir / "channels" / "feishu_threads.json"

    @property
    def feishu_message_store_path(self) -> Path:
        return self.data_dir / "channels" / "feishu_messages.sqlite3"

    @property
    def daily_brief_feishu_chat_id_list(self) -> list[str]:
        return [item.strip() for item in self.daily_brief_feishu_chat_ids.split(",") if item.strip()]

    @property
    def resolved_daily_brief_source_file(self) -> Path:
        path = self.daily_brief_source_file
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def resolved_daily_brief_prompt_file(self) -> Path:
        path = self.daily_brief_prompt_file
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def daily_brief_state_path(self) -> Path:
        return self.data_dir / "channels" / "daily_brief_state.json"

    @property
    def daily_quiz_data_dir(self) -> Path:
        return self.data_dir / "study" / "shenlun"

    @property
    def daily_quiz_scheduler_state_path(self) -> Path:
        return self.daily_quiz_data_dir / "scheduler_state.json"

    @property
    def resolved_daily_brief_api_key(self) -> str:
        return self.daily_brief_api_key or self.memory_api_key or self.api_key

    @property
    def resolved_daily_brief_base_url(self) -> str:
        return self.daily_brief_base_url or self.memory_base_url or self.base_url

    @property
    def wecom_allowed_user_id_set(self) -> set[str]:
        return {item.strip() for item in self.wecom_allowed_user_ids.split(",") if item.strip()}

    @property
    def wecom_user_id_mapping(self) -> dict[str, str]:
        try:
            value = json.loads(self.wecom_user_id_map)
        except (TypeError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            str(wecom_id).strip(): str(user_id).strip()
            for wecom_id, user_id in value.items()
            if str(wecom_id).strip() and str(user_id).strip()
        }

    @property
    def wecom_thread_store_path(self) -> Path:
        return self.data_dir / "channels" / "wecom_threads.json"

    @property
    def wecom_rpa_allowed_group_set(self) -> set[str]:
        return {item.strip() for item in self.wecom_rpa_allowed_groups.split(",") if item.strip()}

    @property
    def wecom_rpa_trigger_prefix_list(self) -> list[str]:
        return [item.strip() for item in self.wecom_rpa_trigger_prefixes.split(",") if item.strip()]

    @property
    def wecom_rpa_state_path(self) -> Path:
        return self.data_dir / "channels" / "wecom_rpa_state.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
