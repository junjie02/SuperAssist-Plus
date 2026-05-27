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
    max_tool_calls: int = Field(default=8, alias="SUPERASSIST_PLUS_MAX_TOOL_CALLS")
    enable_tools: bool = Field(default=False, alias="SUPERASSIST_PLUS_ENABLE_TOOLS")
    memory_llm_writer_enabled: bool = Field(default=False, alias="SUPERASSIST_PLUS_MEMORY_LLM_WRITER_ENABLED")

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
    def huggingface_cache_dir(self) -> Path:
        return self.data_dir / "huggingface"

    @property
    def faiss_dir(self) -> Path:
        return self.data_dir / "faiss"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
