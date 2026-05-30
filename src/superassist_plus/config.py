from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for SuperAssist-Plus."""

    model_provider: str = Field(default="openai", alias="SUPERASSIST_PLUS_MODEL_PROVIDER")
    model: str = Field(default="gpt-4o-mini", alias="SUPERASSIST_PLUS_MODEL")
    api_key: str = Field(default="", alias="SUPERASSIST_PLUS_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="SUPERASSIST_PLUS_BASE_URL")
    temperature: float | None = Field(default=None, alias="SUPERASSIST_PLUS_TEMPERATURE")
    max_tokens: int | None = Field(default=None, alias="SUPERASSIST_PLUS_MAX_TOKENS")
    data_dir: Path = Field(default=Path(".superassist-plus"), alias="SUPERASSIST_PLUS_DATA_DIR")
    tool_workspace_dir: Path | None = Field(default=None, alias="SUPERASSIST_PLUS_TOOL_WORKSPACE_DIR")
    tool_network_enabled: bool = Field(default=True, alias="SUPERASSIST_PLUS_TOOL_NETWORK_ENABLED")
    tool_shell_enabled: bool = Field(default=False, alias="SUPERASSIST_PLUS_TOOL_SHELL_ENABLED")
    tool_shell_timeout_seconds: int = Field(default=120, alias="SUPERASSIST_PLUS_TOOL_SHELL_TIMEOUT_SECONDS")
    tool_shell_output_max_chars: int = Field(default=20000, alias="SUPERASSIST_PLUS_TOOL_SHELL_OUTPUT_MAX_CHARS")
    max_tool_calls: int = Field(default=8, alias="SUPERASSIST_PLUS_MAX_TOOL_CALLS")
    enable_tools: bool = Field(default=False, alias="SUPERASSIST_PLUS_ENABLE_TOOLS")
    subagents_enabled: bool = Field(default=True, alias="SUPERASSIST_PLUS_SUBAGENTS_ENABLED")
    subagent_max_concurrent: int = Field(default=3, alias="SUPERASSIST_PLUS_SUBAGENT_MAX_CONCURRENT")
    subagent_timeout_seconds: int = Field(default=900, alias="SUPERASSIST_PLUS_SUBAGENT_TIMEOUT_SECONDS")
    subagent_max_turns: int = Field(default=20, alias="SUPERASSIST_PLUS_SUBAGENT_MAX_TURNS")
    memory_llm_writer_enabled: bool = Field(default=False, alias="SUPERASSIST_PLUS_MEMORY_LLM_WRITER_ENABLED")
    short_memory_token_limit: int = Field(default=80000, alias="SUPERASSIST_PLUS_SHORT_MEMORY_TOKEN_LIMIT")
    short_memory_keep_recent_turns: int = Field(default=10, alias="SUPERASSIST_PLUS_SHORT_MEMORY_KEEP_RECENT_TURNS")
    short_memory_summary_target_tokens: int = Field(
        default=6000,
        alias="SUPERASSIST_PLUS_SHORT_MEMORY_SUMMARY_TARGET_TOKENS",
    )
    short_memory_enable_tool_events: bool = Field(
        default=True,
        alias="SUPERASSIST_PLUS_SHORT_MEMORY_ENABLE_TOOL_EVENTS",
    )

    memory_reinforce_similarity: float = Field(default=0.85, alias="SUPERASSIST_PLUS_MEMORY_REINFORCE_SIMILARITY")
    memory_concept_merge_similarity: float = Field(default=0.85, alias="SUPERASSIST_PLUS_MEMORY_CONCEPT_MERGE_SIMILARITY")
    memory_completion_similarity: float = Field(default=0.30, alias="SUPERASSIST_PLUS_MEMORY_COMPLETION_SIMILARITY")
    memory_completion_top_k: int = Field(default=5, alias="SUPERASSIST_PLUS_MEMORY_COMPLETION_TOP_K")
    memory_debounce_seconds: float = Field(default=30.0, alias="SUPERASSIST_PLUS_MEMORY_DEBOUNCE_SECONDS")
    memory_decay_lambda: float = Field(default=0.005, alias="SUPERASSIST_PLUS_MEMORY_DECAY_LAMBDA")
    memory_edge_delete_threshold: float = Field(default=0.15, alias="SUPERASSIST_PLUS_MEMORY_EDGE_DELETE_THRESHOLD")
    memory_top_k: int = Field(default=12, alias="SUPERASSIST_PLUS_MEMORY_TOP_K")
    memory_candidate_pool_size: int = Field(default=150, alias="SUPERASSIST_PLUS_MEMORY_CANDIDATE_POOL_SIZE")
    memory_read_use_ppr: bool = Field(default=True, alias="SUPERASSIST_PLUS_MEMORY_READ_USE_PPR")
    memory_read_entry_points: int = Field(default=10, alias="SUPERASSIST_PLUS_MEMORY_READ_ENTRY_POINTS")
    memory_read_max_depth: int = Field(default=3, alias="SUPERASSIST_PLUS_MEMORY_READ_MAX_DEPTH")
    memory_read_bfs_weight: float = Field(default=0.6, alias="SUPERASSIST_PLUS_MEMORY_READ_BFS_WEIGHT")
    memory_read_ppr_weight: float = Field(default=0.4, alias="SUPERASSIST_PLUS_MEMORY_READ_PPR_WEIGHT")
    memory_read_bfs_decay: float = Field(default=0.7, alias="SUPERASSIST_PLUS_MEMORY_READ_BFS_DECAY")
    embedding_provider: str = Field(default="bge", alias="SUPERASSIST_PLUS_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-base-zh-v1.5", alias="SUPERASSIST_PLUS_EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="SUPERASSIST_PLUS_EMBEDDING_DEVICE")
    feishu_app_id: str = Field(default="", alias="SUPERASSIST_PLUS_FEISHU_APP_ID")
    feishu_app_secret: str = Field(default="", alias="SUPERASSIST_PLUS_FEISHU_APP_SECRET")
    feishu_domain: str = Field(default="https://open.feishu.cn", alias="SUPERASSIST_PLUS_FEISHU_DOMAIN")
    feishu_allowed_open_ids: str = Field(default="", alias="SUPERASSIST_PLUS_FEISHU_ALLOWED_OPEN_IDS")
    feishu_mention_only: bool = Field(default=True, alias="SUPERASSIST_PLUS_FEISHU_MENTION_ONLY")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "superassist_plus.sqlite3"

    @property
    def resolved_tool_workspace_dir(self) -> Path:
        return self.tool_workspace_dir or self.data_dir / "workspace"

    @property
    def huggingface_cache_dir(self) -> Path:
        return self.data_dir / "huggingface"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"

    @property
    def feishu_allowed_open_id_set(self) -> set[str]:
        return {item.strip() for item in self.feishu_allowed_open_ids.split(",") if item.strip()}

    @property
    def feishu_thread_store_path(self) -> Path:
        return self.data_dir / "channels" / "feishu_threads.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
