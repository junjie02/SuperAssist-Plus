# superassist_plus Package Technical Documentation

IMPORTANT: Any change to this package's public behavior, exported modules,
configuration, shared models, LLM factory, CLI, or package-level ownership must
update this document.

## Purpose

`superassist_plus` is the main Python package for the LangGraph/LangChain
rewrite of SuperAssist. It owns the runtime shell, shared domain models, model
factory, command-line interface, tools, agent graph, and memory system.

## Key Files

- `__init__.py`: package metadata and version.
- `config.py`: `Settings` model and cached `get_settings()` helper.
- `models.py`: shared Pydantic models, memory node/edge ontology, and run
  result model.
- `llm.py`: model factory for OpenAI-compatible chat models, MiniMax
  compatibility adapter, and local fallback.
- `cli.py`: command-line entrypoint for single-turn and interactive operation.
- `agent/`: LangGraph runtime and LangChain middleware.
- `memory/`: CogniFold-style long-term memory with FAISS dense retrieval.
- `tools/`: LangChain tool adapters.
- `ui/`: FastAPI server for the memory graph viewer.

## Configuration

Settings are loaded through `pydantic-settings`. The `.env` file is resolved
from the project root, not the shell's current working directory. Environment
variables use the `SUPERASSIST_PLUS_` prefix. The important runtime values are:

- `SUPERASSIST_PLUS_MODEL_PROVIDER`
- `SUPERASSIST_PLUS_MODEL`
- `SUPERASSIST_PLUS_API_KEY`
- `SUPERASSIST_PLUS_BASE_URL`
- `SUPERASSIST_PLUS_TEMPERATURE`
- `SUPERASSIST_PLUS_MAX_TOKENS`
- `SUPERASSIST_PLUS_DATA_DIR`
- `SUPERASSIST_PLUS_ENABLE_TOOLS`
- `SUPERASSIST_PLUS_MEMORY_LLM_WRITER_ENABLED`
- memory thresholds and debounce settings
- `SUPERASSIST_PLUS_EMBEDDING_PROVIDER`
- `SUPERASSIST_PLUS_EMBEDDING_MODEL`
- `SUPERASSIST_PLUS_EMBEDDING_DEVICE`

`Settings.db_path` derives the SQLite database path from `data_dir`.

## LLM Behavior

`create_chat_model()` returns:

- `ChatOpenAI` when an API key is configured.
- `MiniMaxCompatibleChatModel` for MiniMax model names or MiniMax base URLs.
- `FallbackChatModel` when no API key is configured.

The fallback model supports `bind_tools()` so LangChain `create_agent` can run
without provider credentials. It never calls tools and returns deterministic
text for local tests.

## Embedding Behavior

Memory embeddings are configurable. The default provider is BGE:

- `SUPERASSIST_PLUS_EMBEDDING_PROVIDER=bge`
- `SUPERASSIST_PLUS_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5`
- `SUPERASSIST_PLUS_EMBEDDING_DEVICE=cpu`

The hash provider is retained only as a deterministic local fallback for tests
or offline development.

Persisted node embeddings are indexed in FAISS files under
`{SUPERASSIST_PLUS_DATA_DIR}/faiss/`, with mapping JSON files from FAISS integer
ids to memory node ids. SQLite `memory_nodes.embedding_json` keeps an
authoritative copy for rebuilding and audit.

## CLI Behavior

`superassist-plus "message"` runs one turn.

`superassist-plus --interactive` starts a continuous REPL using one thread id.

`--flush-memory` forces queued memory writes to complete before process exit.

## Tool Calling

Tool calling is controlled by `SUPERASSIST_PLUS_ENABLE_TOOLS`. It defaults to
`false` because some OpenAI-compatible providers, including MiniMax-style chat
endpoints, may reject tool schemas with provider-specific `invalid chat setting`
errors. Enable it only when the selected model endpoint supports OpenAI-style
tool calling.

For MiniMax model names, the LLM factory defaults `temperature` to `1.0`,
matching the ProAssist configuration note that MiniMax requires temperature in
`(0.0, 1.0]`.

MiniMax compatibility keeps LangChain's `create_agent` OpenAI-style tool
binding enabled and adds `extra_body.reasoning_split=true`, matching MiniMax's
OpenAI-compatible tool/function-calling documentation.

## Memory Graph UI

`superassist-plus-memory-ui` starts a FastAPI/uvicorn server for the static
frontend and read-only graph API.
It visualizes SQLite memory nodes and typed edges and shows a right-side update
ledger derived from node/edge `updated_at` timestamps.

## Memory Writer Mode

Memory writes are deterministic by default. `SUPERASSIST_PLUS_MEMORY_LLM_WRITER_ENABLED`
defaults to `false` so regular chat models are not asked to emit strict JSON on
every turn. Enable it only after selecting a model/prompt combination that is
known to return valid memory plans.

## Maintenance Notes

- Keep package-level imports small and stable.
- Shared domain types belong in `models.py`; module-specific types should stay
  in their module.
- If a new runtime subsystem is added, create a subdirectory-level `AGENT.md`.
