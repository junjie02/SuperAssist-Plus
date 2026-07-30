# src/ 技术方案

本文档面向 `src/superassist/` 包内部，按"模块 → 类/函数 → 字段"三层粒度记录 SuperAssist 的实现规约。项目根 [CLAUDE.md](../CLAUDE.md) 是"在哪里找什么"的导航；本文是"每个数据结构每一列代表什么、谁写谁读"的契约。

源码统一遵循以下约定：

- Python ≥ 3.11，启用 `from __future__ import annotations`。
- 所有可观察数据结构使用 Pydantic v2 (`BaseModel`) 或 `@dataclass(frozen=True)`。可变运行态（队列、缓存、会话池）使用 `@dataclass`（非 frozen）。
- 所有时间戳为 UTC，写库使用 `datetime.fromisoformat()` 反序列化、`isoformat()` 序列化。
- 配置在 [`config.Settings`](superassist/config.py) 集中，环境变量前缀 `SUPERASSIST_`。
- 没有外层 LangGraph 包装：lead agent = 单次 `langchain.agents.create_agent` + 一条确定性 middleware 链。子 agent (`SubagentExecutor`) 才使用 `StateGraph`。

---

## 目录

1. [`models.py` — 内存图本体与运行结果](#1-modelspy)
2. [`config.py` — Settings 全量字段](#2-configpy)
3. [`llm.py` — Chat 模型工厂与 MiniMax 适配](#3-llmpy)
4. [`observability.py` / `run_events.py` — Trace 与运行事件总线](#4-observability--run_events)
5. [`memory/` — CogniFold 类型图记忆](#5-memory)
6. [`agent/` — runtime / factory / state / prompts / streaming / short_memory](#6-agent)
7. [`middlewares/` — 9 条基础 middleware + 3 条 RAG middleware](#7-middlewares)
8. [`subagents/` — in-process `task` 子智能体](#8-subagents)
9. [`acp_client/` — ACP 协议客户端](#9-acp_client)
10. [`teams/` — `team_task` 监督器与哈希链 ledger](#10-teams)
11. [`tools/` — LangChain `@tool` 集合](#11-tools)
12. [`skills/` — SKILL.md 注册表与虚拟路径](#12-skills)
13. [`channels/feishu` / `channels/wecom` — 飞书与企业微信通道](#13-channelsfeishu)
14. [`ui/server.py` — FastAPI 记忆图查看器后端](#14-uiserverpy)
15. [`cli.py` — `superassist` 命令入口](#15-clipy)
16. [跨模块时序：一条用户消息从进入到落库](#16-cross-module-sequence)
17. [`rag/` — LightRAG 文档知识库与 Agentic RAG](#17-rag)

---

<a id="1-modelspy"></a>
## 1. `models.py`

[superassist/models.py](superassist/models.py) 定义全包共享的核心枚举与 Pydantic 模型。**修改这里属于不向后兼容变更**，必须同步更新 [`tests/test_smoke.py`](../tests/test_smoke.py) 与 [`memory/storage.py`](superassist/memory/storage.py) 的 schema。

### 1.1 `NodeType` (str Enum)

| 值 | 语义 | 写入主体 |
| --- | --- | --- |
| `event` | 一次用户/助手 turn 的原始事件节点 | runtime 预分配 ID，LLM/fallback Memory writer 通过 `ADD_NODE(ref="current_event")` 创建。 |
| `concept` | 反复出现的稳定模式、用户偏好、复用上下文 | LLM writer / fallback writer |
| `intent` | 未达成目标、待办、跟进项 | LLM writer |
| `time` | 截止时间或时间锚点 | LLM writer；其 embedding 强制为 `None`（见 `_embed_for`）。 |

### 1.2 `EdgeType` (str Enum) 与默认权重

`EDGE_TYPE_DEFAULT_WEIGHTS` 决定 `add_or_boost_edge(weight=None)` 时的初始 weight：

| EdgeType | 默认权重 | 允许的 source / target 组合 (`EDGE_TYPE_CONSTRAINTS`) | 语义 |
| --- | --- | --- | --- |
| `GROUNDS` | 0.9 | event → concept/intent | 直接证据 |
| `CAUSES` | 0.9 | event → event | 因果链 |
| `TRIGGERS` | 0.8 | event/concept → intent | 激活目标 |
| `USER_FEEDBACK` | 0.8 | event/concept → intent | 显式纠正/偏好 |
| `REINFORCES` | 0.7 | event → concept | 已有概念的支持证据 |
| `PART_OF` | 0.7 | concept → concept | 层级/包含 |
| `DERIVED_FROM` | 0.6 | concept → concept | 抽象/派生 |
| `DEADLINE_FOR` | 0.6 | time → event/concept/intent | 时间约束 |
| `RELATED_TO` | 0.5 | concept → concept | 兜底关联 |

`MemoryGraphStore._validate_edge` 会按此表拒绝越界连接，并在 `add_or_boost_edge` 把 `weight` 强制 clip 到 `[0.0, 1.0]`。

### 1.3 `MemoryNode`

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `id` | `str` | — | `new_id(node_type.value)` 生成，形如 `concept_3f7a8b...`（12 hex）。 |
| `user_id` | `str` | — | 隔离单位；`MemoryService` 的所有读路径都按 `user_id` 过滤。 |
| `type` | `NodeType` | — | 见 §1.1。 |
| `title` | `str` | — | 写库前会被 `add_node` 截断为非空（`title.strip() or node_type.value`）。 |
| `description` | `str` | — | 持久化的"可复用记忆"；`add_node` 调 `description.strip()`。 |
| `importance` | `float` | `0.5` | LLM writer 可写；`UpdateNodeOp` 视 `0.5` 为"未变更"哨兵。 |
| `access_count` | `int` | `0` | `touch_nodes` 每次 +1，作为 ranker 的 `access` 维度分母。 |
| `embedding` | `list[float] \| None` | `None` | TIME 节点强制 None；其余节点写库时 `json.dumps`。 |
| `reasoning` | `str` | `""` | LLM 解释为何创建该节点；UI 优先展示 `reasoning` 而非 `description`。 |
| `grounded_in` | `list[str]` | `[]` | 节点级 grounding（除运行时根据 `_GROUNDING_RULES` 自动加边外，UI 也会读这个数组）。 |
| `metadata` | `dict[str, Any]` | `{}` | 至少包含 `source` / `thread_id` / `plan_format`；运行时回填 `merged: True` 标记 MERGE_NODES 留下的"幸存者"。 |
| `created_at` / `updated_at` | `datetime` | `utc_now()` | 任何 update 都会刷 `updated_at`。 |
| `last_accessed_at` | `datetime \| None` | `None` | 由 `touch_nodes` 写入；`recency` 计算的 anchor。 |

### 1.4 `MemoryEdge`

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `id` | `str` | — | `new_id("edge")`。 |
| `user_id` | `str` | — | 隔离单位。 |
| `source_id` / `target_id` | `str` | — | 必须是同 `user_id` 下存在的节点；FK 在 SQLite 上为 `ON DELETE CASCADE`。 |
| `edge_type` | `EdgeType` | — | 受 `EDGE_TYPE_CONSTRAINTS` 约束。 |
| `weight` | `float` | — | 0.0–1.0；`add_or_boost_edge` 命中已存在边时 `min(1.0, w + boost=0.05)`。 |
| `metadata` | `dict[str, Any]` | `{}` | 关键字段：`source`（`memory_writer` / 自动 grounding）、`mechanic`（`grounded_in` / `accumulation` / `completion`）、`similarity`、`reasoning`。 |
| `created_at` / `updated_at` / `last_activated_at` | `datetime` / `datetime` / `datetime?` | `utc_now()` / `utc_now()` / `None` | `last_activated_at` 在 `add_or_boost_edge` 命中现有边时刷新。 |

唯一约束：`UNIQUE(user_id, source_id, target_id, edge_type)` —— 同一对节点同一类型只允许一条边，重复写入走 boost 路径。

### 1.5 `MemoryRecall`

四桶节点列表，被 `DynamicContextMiddleware` 序列化为 JSON 注入 system prompt：

| 桶 | 来源 |
| --- | --- |
| `immediate` | tier1（最相关），固定 1 条（当 `limit > 0`）。 |
| `working` | tier2，`int(limit * 0.30)` 条。 |
| `background` | tier3，`int(limit * 0.50)` 条。 |
| `buffer` | 兜底，填满 `limit - 已选`。 |

### 1.6 `AgentRunResult` / `AgentRunEvent`

`AgentRuntime.run*` 返回 `AgentRunResult`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `thread_id` | `str` | 实际使用的 thread id（如未传入则为 `thread_<12hex>`）。 |
| `answer` | `str` | 来自 `metadata["final_assistant_text"]` 或最后一条 `AIMessage`。 |
| `metadata` | `dict[str, Any]` | 关键键见 §6.1。 |

`AgentRunEvent`（运行事件总线消息体）：

| 字段 | 说明 |
| --- | --- |
| `type` | `preparing_context` / `thinking` / `agent_text` / `subagent_text` 之一。 |
| `message` | 已剥首尾空白的可展示文本。 |
| `metadata` | 通常带 `thread_id`；`subagent_text` 还带 `task_id` / `description` / `subagent_type`。 |

---

<a id="2-configpy"></a>
## 2. `config.py`

[superassist/config.py](superassist/config.py) 通过 `pydantic_settings.BaseSettings` 暴露所有可调参数。`get_settings()` 用 `lru_cache(maxsize=1)` 缓存单例，加载顺序：环境变量 → `PROJECT_ROOT/.env` → 字段默认值。`PROJECT_ROOT` 永远等于 `Path(__file__).resolve().parents[2]`，即仓库根目录（不是 `src/`）。

### 2.1 字段对照表（按用途分组）

**模型**

| 字段 | 别名 ENV | 默认 | 备注 |
| --- | --- | --- | --- |
| `model_provider` | `SUPERASSIST_MODEL_PROVIDER` | `openai` | 当前唯一支持的 provider；其它值在 `create_chat_model` 抛 `ValueError`。 |
| `model` | `SUPERASSIST_MODEL` | `gpt-4o-mini` | 字符串包含 `minimax` 时走 `MiniMaxCompatibleChatModel` 分支。 |
| `api_key` | `SUPERASSIST_API_KEY` | `""` | **空字符串** → `FallbackChatModel`（确定性本地回声）。 |
| `base_url` | `SUPERASSIST_BASE_URL` | `https://api.openai.com/v1` | 可指向兼容 OpenAI 协议的网关。 |
| `temperature` | `SUPERASSIST_TEMPERATURE` | `None` | 模型名含 `minimax` 且未显式设置时强制为 `1.0`。 |
| `reasoning_effort` | `SUPERASSIST_REASONING_EFFORT` | `medium` | GPT-5.6 支持 `none/low/medium/high/xhigh/max`；飞书 `/effort` 可按会话覆盖。 |
| `max_tokens` | `SUPERASSIST_MAX_TOKENS` | `None` | 仅当非 None 才透传。 |
| `model_input_log_enabled` | `SUPERASSIST_MODEL_INPUT_LOG_ENABLED` | `False` | 开启后记录最终 provider payload 到 `<data_dir>/logs/model-input.jsonl`。 |
| `model_input_log_max_bytes` | `SUPERASSIST_MODEL_INPUT_LOG_MAX_BYTES` | `52428800` | 单日志文件轮转阈值；保留 `.1` 到 `.3` 三个备份。 |

**工具**

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `tool_workspace_dir` | `None` | 为空时取 `data_dir/workspace`（见 `resolved_tool_workspace_dir`）。 |
| `tool_network_enabled` | `True` | 关闭后 `web_search` / `web_fetch` 直接返回错误串。 |
| `tool_shell_enabled` | `False` | 默认禁用 shell 工具。 |
| `tool_shell_timeout_seconds` | `120` | `shell` 工具的硬上限（写入时还会再 clamp 到 `[1, 600]`）。 |
| `tool_shell_output_max_chars` | `20000` | 超长时按 `_truncate` 中段替换为 `... [truncated N chars] ...`。 |
| `max_tool_calls` | `8` | `ToolCallLimitMiddleware` 的每 turn 预算（按 `tool_result` 计数）。 |
| `enable_tools` | `False` | **总开关**：False 时 `default_tools()` 返回空，子 agent / team 都不挂载。 |

**子智能体**

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `subagents_enabled` | `True` | 控制 `task` 工具是否挂载，以及 `SubagentLimitMiddleware` 是否进入链。 |
| `subagent_max_concurrent` | `3` | `SubagentLimitMiddleware` 强制 clamp 到 `[1,3]`；`tools/task.py` 的 `BoundedSemaphore` 同样为 3。**两侧必须同步**。 |
| `subagent_timeout_seconds` | `900` | 单次子任务硬超时；`asyncio.timeout(...)` 触发 `TIMED_OUT`。 |
| `subagent_max_turns` | `20` | LangGraph `recursion_limit` 上限；超出走 `_summarize_after_recursion_limit`。 |

**记忆**

| 字段 | 默认 | 用处 |
| --- | --- | --- |
| `memory_llm_writer_enabled` | `True` | True 时用独立 Memory Updater 走 `MEMORY_WRITER_PROMPT`，否则 fallback writer。 |
| `memory_model` | `deepseek-v4-flash` | Memory Updater 与短记忆压缩使用的独立 OpenAI 兼容模型。 |
| `memory_api_key` / `memory_base_url` | 空 | 独立模型凭据与地址；为空时复用主模型配置。 |
| `short_memory_token_limit` | `80000` | 活动段达到该 token 阈值后整体压缩为摘要。 |
| `short_memory_keep_recent_turns` | `30` | 活动段达到该已完成回合数后整体压缩为摘要。 |
| `short_memory_summary_target_tokens` | `6000` | 压缩 prompt 中告知模型的目标长度。 |
| `short_memory_enable_tool_events` | `False` | 已废弃的兼容字段；短记忆始终只保存 user 与最终 assistant。 |
| `memory_reinforce_similarity` | `0.85` | event ↔ 现有 concept 余弦相似度阈值；命中即 `REINFORCES`。 |
| `memory_concept_merge_similarity` | `0.85` | `merge_similar_concepts` 阈值。 |
| `memory_completion_similarity` | `0.30` | `complete_orphans` 给孤立 concept 接 GROUNDS 的下限。 |
| `memory_completion_top_k` | `5` | 每个孤立 concept 最多对比的候选 event 数。 |
| `memory_debounce_seconds` | `30.0` | `MemoryWriteQueue` 的 timer。 |
| `memory_decay_lambda` | `0.005` | 边权 `weight * exp(-λ * age_days)`。 |
| `memory_edge_delete_threshold` | `0.15` | 衰减后低于阈值即删除。 |
| `memory_top_k` | `12` | 每次 recall 选入 prompt 的总节点数（四桶之和）。 |
| `memory_candidate_pool_size` | `150` | ranker 每次只在分数最高的前 N 名里挑桶。 |
| `memory_read_use_ppr` | `True` | 读路径混入 Personalized PageRank。 |
| `memory_read_entry_points` | `10` | FAISS 给 BFS / PPR 注入的 entry point 数量。 |
| `memory_read_max_depth` | `3` | BFS 最大跳数。 |
| `memory_read_bfs_weight` / `memory_read_ppr_weight` | `0.6` / `0.4` | 二者归一化后混合 BFS 与 PPR。 |
| `memory_read_bfs_decay` | `0.7` | BFS 每跳衰减系数。 |

**Embedding**

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `embedding_provider` | `bge` | `bge` → `BGEEmbedder`；`hash` / `local` / `fallback` → `HashEmbedder`（256 维）。 |
| `embedding_model` | `BAAI/bge-base-zh-v1.5` | 仅 BGE 使用。 |
| `embedding_device` | `cpu` | sentence-transformers `device` 参数。 |

**LightRAG 知识库**

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `rag_max_file_size_mb` | `25` | Python 单文件大小上限；Go 代理还会按批次数量计算请求体上限。 |
| `rag_max_files_per_batch` | `20` | 单次 multipart 上传文件数上限。 |
| `rag_max_attempts` | `3` | 一轮 Agentic RAG 中 `RagTurnSession` 允许的检索次数。 |
| `rag_top_k` | `20` | LightRAG 实体/关系向量候选数。 |
| `rag_chunk_top_k` | `10` | 原文 chunk 向量候选数。 |
| `rag_context_max_chars` | `24000` | 结构化检索结果注入 Agent 前的字符硬上限。 |

**飞书**

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `feishu_app_id` / `feishu_app_secret` | `""` | 任意为空 → 启动 `FeishuChannel.start` 抛 `RuntimeError`。 |
| `feishu_domain` | `https://open.feishu.cn` | Lark Client 的 domain。 |
| `feishu_allowed_open_ids` | `""` | 逗号分隔；`feishu_allowed_open_id_set` 属性返回去空白后的 set。 |
| `feishu_mention_only` | `True` | True 且非私聊时只响应被 @ 消息。 |
| `feishu_image_ocr_enabled` | `True` | 是否运行本地 RapidOCR；OCR 失败不阻止原图进入主模型。 |
| `feishu_image_ocr_max_chars` | `12000` | 多图 OCR 注入当前轮文本历史的总字符上限。 |
| `feishu_image_context_ttl_seconds` | `180` | 最新一组原图跨消息保留的滑动 TTL；有效后续消息刷新计时。 |

**企业微信**

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `wecom_bot_id` / `wecom_bot_secret` | `""` | 官方智能机器人长连接凭据；任意为空时通道拒绝启动。 |
| `wecom_allowed_user_ids` | `""` | 逗号分隔 userid 白名单；留空允许机器人可见范围内所有成员。 |
| `wecom_user_id_map` | `{}` | JSON 对象；单聊键为 userid，群聊键为 `chat:<chatid>`，映射到 Go/JWT user_id 以共享 Memory 与 LightRAG。 |
| `wecom_rag_mode_default` | `False` | 新会话的知识库检索默认值，后续可通过 `/rag on/off` 覆盖。 |
| `wecom_max_concurrent` | `3` | 企业微信通道全局 Agent 并发上限。 |
| `wecom_stream_interval_ms` | `300` | 向企业微信合并发送流式更新的最小间隔。 |
| `wecom_ai_engine_url` | `http://127.0.0.1:8765` | 复用现有 AI Engine 的地址。 |

### 2.2 派生属性

| 属性 | 计算 |
| --- | --- |
| `db_path` | `data_dir / "superassist.sqlite3"` |
| `resolved_tool_workspace_dir` | `tool_workspace_dir or data_dir / "workspace"` |
| `huggingface_cache_dir` | `data_dir / "huggingface"` |
| `faiss_dir` | `data_dir / "faiss"` |
| `rag_dir` | `data_dir / "rag"` |
| `feishu_thread_store_path` | `data_dir / "channels" / "feishu_threads.json"` |
| `wecom_thread_store_path` | `data_dir / "channels" / "wecom_threads.json"` |

`PROJECT_ROOT` 同时被 [`teams/config.py`](superassist/teams/config.py) 用来定位 `agent_team.toml`、被 [`skills/registry.py`](superassist/skills/registry.py) 定位 `skills/`、被 [`tools/shell.py`](superassist/tools/shell.py) 限制 `cwd` 不可越界、被 [`ui/server.py`](superassist/ui/server.py) 定位 `frontend/`。

---

<a id="3-llmpy"></a>
## 3. `llm.py`

[superassist/llm.py](superassist/llm.py) 提供主模型工厂 `create_chat_model(settings)` 与独立记忆模型工厂 `create_memory_model(settings, call_kind=...)`：

1. **空 API key** → `FallbackChatModel`（不调用网络）。
   - `bind_tools()` 接受参数但永远不调用工具。
   - 检测到 system prompt 含 `"compress conversation history"` 关键字时，走 `_fallback_summary` 把最近 user 消息提炼为 Markdown summary，使短记忆压缩在离线环境下也能跑。
2. **包含 `minimax`（model 名或 base_url 任一）** → `MiniMaxCompatibleChatModel`：
   - `_get_request_payload`：把 OpenAI 风格的 `max_completion_tokens` 改回 `max_tokens`；剥掉每条 message 的 `name` 字段（MiniMax 不接受）；强制 `extra_body.reasoning_split=True`；可选 `SUPERASSIST_DEBUG_MINIMAX_PAYLOAD` 路径转储 payload。
   - `_create_chat_result`：从响应的 `choices[].message.reasoning_details[].text` 抽取 reasoning，与 `<think>...</think>` 内联块合并写入 `AIMessage.additional_kwargs.reasoning_content`。
3. **GPT-5.6 family** → `OneSecondRetryChatModel` + Responses API：
   - 请求 `reasoning.effort`；非 `none` 时同时请求 `reasoning.summary="detailed"`，供流式飞书卡片展示。
   - 开启 `stream_usage`，`AgentRuntime` 把 `input_tokens/cache_read/cache_hit_rate` 写入结果 metadata 和日志。
4. **其它 OpenAI 兼容** → `OneSecondRetryChatModel`，每次 `_generate` 失败后 `time.sleep(1)` 重试一次。

`create_memory_model` 默认使用 `deepseek-v4-flash`，从 `memory_api_key/memory_base_url` 取独立配置（空值回退主模型凭据），不附加 GPT-5.6 的 Responses/reasoning 参数。Memory Updater 与短记忆压缩各持有一个该模型客户端。

所有真实模型请求都可写入 `logs/model-input.jsonl`；记录包含 `call_kind`、近似 token 总量、按 static system / short memory / turn context / current user / tool schemas 等划分的 `input_manifest`，并按配置大小轮转。

`is_minimax_model(model, base_url)` 是上述路由的判定函数；`AgentRuntime` 也用它向 trace 注入 `tool_schema_binding="openai_compatible_minimax"`。

`temperature` 默认 None，但 `model` 含 `minimax` 时强制 `1.0`。GPT-5.6 默认推理强度为 `medium`。

---

<a id="4-observability--run_events"></a>
## 4. `observability.py` / `run_events.py`

### 4.1 `observability.traceable`

LangSmith 可选依赖：导入失败时 `traceable` 退化为透明装饰器，调用方代码无需改动。所有 `runnable_trace_config` 都返回固定形状：

```python
{
  "run_name": str,
  "tags": ["superassist", *extra_tags],
  "metadata": _compact_metadata(metadata),  # 见下方
}
```

`_compact_metadata` 是防爆的 trace 序列化器：字符串裁到 500 字符、列表/字典各取前 20 项、嵌套字符串裁到 200。`without_self` 用作 `process_inputs` 钩子，把绑定方法的 `self` 从 trace inputs 中剔除。

### 4.2 `run_events.RunEventReporter`

```python
RunEventReporter = Callable[[AgentRunEvent], None]
```

通过 `ContextVar` (`_current_run_event_reporter`) 在线程内传播。`run_event_reporter_context(reporter)` 是上下文管理器；`AgentRuntime.run*` 在每次 turn 进入时 `set` 一次，在 `finally` 中自动 `reset`。子 agent / `task` 工具用 `current_run_event_reporter()` 透明地拿到同一个 reporter，避免显式参数透传。

---

<a id="5-memory"></a>
## 5. `memory/`

记忆系统自上而下：`MemoryService` ⇒ `MemoryGraphStore` (SQLite) + `Embedder` + `PersistentFaissIndex` + `MemoryContextRanker`。写路径再加一层 `MemoryWriteQueue` 防抖 + `MemoryWriter`（LLM/fallback）+ `apply_plan`（纯函数）。

### 5.1 `storage.MemoryGraphStore` —— SQLite schema

`init_schema` 启用 `foreign_keys=ON`、`busy_timeout=30s`，创建以下表：

#### 5.1.1 `memory_nodes`

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT PK | 与 `MemoryNode.id` 同形。 |
| `user_id` | TEXT NOT NULL | 隔离键，建有 `idx_memory_nodes_user_type(user_id, type)`。 |
| `type` | TEXT NOT NULL | `NodeType` 字面量。 |
| `title` | TEXT NOT NULL | 写库前已 strip。 |
| `description` | TEXT NOT NULL | 写库前已 strip。 |
| `importance` | REAL NOT NULL DEFAULT 0.5 | |
| `access_count` | INTEGER NOT NULL DEFAULT 0 | `touch_nodes` 自增。 |
| `embedding_json` | TEXT NULL | `json.dumps(list[float])` 或 NULL。 |
| `reasoning` | TEXT NOT NULL DEFAULT '' | |
| `grounded_in_json` | TEXT NOT NULL DEFAULT '[]' | JSON list。 |
| `metadata_json` | TEXT NOT NULL DEFAULT '{}' | JSON dict。 |
| `created_at` / `updated_at` | TEXT NOT NULL | ISO-8601。 |
| `last_accessed_at` | TEXT NULL | |

#### 5.1.2 `memory_edges`

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `id` | TEXT PK | |
| `user_id` | TEXT NOT NULL | 索引：`idx_memory_edges_user_source(user_id, source_id)`、`idx_memory_edges_user_target(user_id, target_id)`。 |
| `source_id` / `target_id` | TEXT NOT NULL | FK → `memory_nodes(id) ON DELETE CASCADE`。 |
| `edge_type` | TEXT NOT NULL | `EdgeType` 字面量。 |
| `weight` | REAL NOT NULL | 写时 clamp 到 `[0,1]`。 |
| `metadata_json` | TEXT NOT NULL DEFAULT '{}' | |
| `created_at` / `updated_at` / `last_activated_at` | TEXT / TEXT / TEXT NULL | |
| `UNIQUE(user_id, source_id, target_id, edge_type)` | | 确保 `add_or_boost_edge` 走 boost 而非重复 INSERT。 |

#### 5.1.3 `memory_jobs`（保留表，目前未写入）

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `id` / `user_id` / `status` / `payload_json` / `attempts` / `error` / `created_at` / `updated_at` | TEXT/TEXT/TEXT/TEXT/INTEGER/TEXT/TEXT/TEXT | 留给未来替换内存队列 `MemoryWriteQueue` 的持久化方案；当前 `MemoryWriter` 不写这张表，`idx_memory_jobs_status(status, updated_at)` 索引已建好。 |

#### 5.1.4 `memory_recall_snapshots`

每次 `prepare_turn_contexts` 都会 **整表 DELETE 后整批 INSERT**（`replace_recall_snapshot`），用作 UI 高亮当前命中的 recall 节点：

| 列 | 类型 |
| --- | --- |
| `user_id` (PK 一部分) | TEXT |
| `node_id` (PK 一部分) | TEXT，FK → `memory_nodes` ON DELETE CASCADE |
| `tier` | TEXT — `immediate` / `working` / `background` / `buffer` |
| `score` | REAL — ranker 给出的最终 `Score(v)` |
| `pagerank` / `recency` / `access` / `urgency` / `semantic_affinity` | REAL —  分项分数，UI 在节点详情面板展示 |
| `updated_at` | TEXT |

#### 5.1.5 关键写方法语义

- `add_node(...)`: 生成 ID（若未传入 `node_id`），strip title/description，序列化 embedding/grounded_in/metadata 后插入；返回 `MemoryNode`。
- `update_node(node)`: 全列更新；不动 `created_at`，刷新 `updated_at`。
- `add_or_boost_edge(...)`: 先 `_validate_edge` 检查类型组合 → 若已存在则 `weight = min(1.0, existing.weight + boost(默认 0.05))`，并合并 metadata；否则按 `EDGE_TYPE_DEFAULT_WEIGHTS[edge_type]` 写入。
- `replace_edge_endpoint(user_id, old_id, new_id)`: MERGE_NODES 时把所有指向 `old_id` 的边重写到 `new_id`，使用 `UPDATE OR IGNORE` 静默吸收唯一约束冲突（即被合并节点的边若与幸存节点的边重复则直接丢弃）。
- `touch_nodes(user_id, ids)`: `access_count + 1` 并刷 `last_accessed_at`、`updated_at`；`prepare_turn_contexts` 在选完读/写桶后调用一次。
- `delete_edges(ids)`: `executemany`，空列表直接返回。
- `replace_recall_snapshot`: 见 5.1.4。

### 5.2 `embedding.py`

`Embedder` Protocol 暴露 `embed(text) -> list[float]` 与 `preload()`。

- `HashEmbedder(dimensions=256)`: 把文本按 `TOKEN_RE`（Unicode 词 + 中日韩字符）切词，每个 token 用 `blake2b(8)` 取前 4 字节模 dimensions 做 index、第 5 字节奇偶定符号；最终 L2 normalize。完全离线、确定性，可直接喂 FAISS（点积即余弦）。
- `BGEEmbedder(model_name, device, cache_dir)`: 懒加载 `sentence_transformers.SentenceTransformer`，`embed` 调用 `encode([text], normalize_embeddings=True)`。`cache_dir` 来自 `Settings.huggingface_cache_dir`。
- `get_embedder(settings)` 走 `_cached_embedder(provider, model, device, cache_dir)` —— `lru_cache(8)` 按 (provider, model, device, cache_dir) 缓存实例。`MemoryService.preload_embedder()` 在 `build_agent` 时强制实例化 BGE 模型，避免首条用户消息时再阻塞。
- `cosine_similarity(a, b)`: 长度不匹配或任一为空时返回 0，否则按位点积；调用方需保证两侧已 L2 normalize（HashEmbedder/BGEEmbedder 都满足）。

### 5.3 `vector_index.PersistentFaissIndex`

每个 `user_id` 一个文件对：`<faiss_dir>/<safe_user_id>.index` + `<safe_user_id>.mapping.json`（`safe_user_id` 把非字母数字字符替换为 `_`）。

- `rebuild(nodes)`：过滤掉无 embedding 或维度不匹配的节点；空时调 `_delete_files()` 删除两侧文件；否则用 `IndexIDMap2(IndexFlatIP(dim))`、`add_with_ids(matrix, np.arange(N))`、写盘并把 `dimension` + `ids: list[str]` 落到 mapping。`MemoryService.apply_structured_memory` 在 nodes/updated/merged/removed_nodes 任一非零时触发重建。
- `search(query, limit)`：mapping 文件不存在或维度不匹配返回 `[]`；否则 normalize 查询向量后 `index.search`，把 FAISS 内部 id 翻译为节点 id，封装为 `VectorMatch(node_id, score)`。
- `EmptyVectorIndex`：测试 / fallback 兜底，所有方法 no-op。

### 5.4 `scoring.MemoryContextRanker`

类常量：`damping=0.85`，`node_decay_lambda_per_hour=0.01`，`urgency_window_hours=24.0`。

#### 5.4.1 `EventProbe`（frozen dataclass）

| 字段 | 含义 |
| --- | --- |
| `user_id` | 隔离键 |
| `text` | 原始查询文本 |
| `embedding` | 已归一化的查询向量 |
| `timestamp` | `recency` / `urgency` 的时间锚（默认 `datetime.now(UTC)`） |

#### 5.4.2 `MemoryNodeScore`（frozen dataclass）

| 字段 | 计算 |
| --- | --- |
| `pagerank` | `compute_pagerank` 或 `compute_personalized_pagerank` 的归一化值 |
| `recency` | `exp(-0.01 * age_hours)`，age 取 `last_accessed_at → updated_at → created_at` 优先级 |
| `access` | `node.access_count / max(access_count over user)`，无访问时为 0 |
| `urgency` | INTENT 节点且有 `DEADLINE_FOR` 边 + 24 小时内的 TIME 节点：`1.0 + (1 - hours/24)`，clamp `[1.0, 2.0]`；其它一律 `1.0` |
| `score` | `(0.4·pagerank + 0.4·recency + 0.2·access) * urgency` |
| `semantic_affinity` | `cosine_similarity(probe.embedding, node.embedding)` |

#### 5.4.3 写路径 `assemble_context`

被 `MemoryService.prepare_turn_contexts` 用于产出 `write_recall`：
1. 全图 `compute_pagerank`（构建 `MultiDiGraph`，边权 = `weight * exp(-λ * age_days)`，无边时退化为 `dict.fromkeys(..., 1.0)`）。
2. `score_nodes` 计算 `MemoryNodeScore`，按 `(score, semantic_affinity)` 倒序取 `memory_candidate_pool_size` 名作为 pool。
3. `_select_tiers(limit)`：
   - `immediate` (1 名)：按 `0.7·recency + 0.3·(urgency-1)` 排序。
   - `working` (`int(limit*0.30)`)：按 `0.5·pagerank + 0.3·recency + 0.2·_type_bonus(node)` 排序，`_type_bonus` 为 CONCEPT 1.0 / INTENT 0.8 / TIME 0.4 / EVENT 0.2。
   - `background` (`int(limit*0.50)`)：`_background_ranked` 按 `0.8·pagerank + 0.2·diversity` 排序，`diversity = 1/同类型已选节点数`，鼓励桶内类型多样。
   - `buffer` (剩余名额)：直接按 `score` 倒序补齐到 `limit`。
   - 全程通过 `selected: set[str]` 去重。

#### 5.4.4 读路径 `assemble_read_context`

被 `MemoryService.prepare_turn_contexts` 用于产出 `read_recall`、被 `MemoryService.recall(query, limit)` 直接使用：

1. 接收 FAISS 给的 `entry_matches[: memory_read_entry_points]`，构造 `entry_scores: dict[node_id → score]`（相同节点取最大）。
2. 若 entry 为空 → 退回 `assemble_context`（写路径）。
3. `_bfs_scores`：以 entry 为种子，无向 BFS，最大跳数 `memory_read_max_depth`，每跳乘以 `memory_read_bfs_decay=0.7`，遇到更高分的路径才更新。
4. 若 `memory_read_use_ppr=True`：`compute_personalized_pagerank` 把 entry 作为 `personalization` 向量；与 BFS 按归一化权重 `0.6:0.4` 混合（见 `_blend_read_scores`）。否则只用 BFS。
5. `score_nodes(probe, nodes, read_rank)` 把混合排名当作 `pagerank` 维度，再走相同的 `_select_tiers` 流程。

#### 5.4.5 时间相关 helper

- `_deadline_from_node(time_node)`：先看 metadata 里的 `scheduled_time` / `deadline` / `datetime` / `time` 字段，再尝试把 `description` / `title` 自身当作 ISO 时间解析；任一成功即返回 `datetime`。
- `_as_utc(dt)`：naive datetime 视作 UTC，否则 `astimezone(UTC)`，避免与 probe 时间比较时报 `TypeError`。

### 5.5 `plans.py` —— `UpdatePlan`

每条 plan 操作都是一个 Pydantic 模型；`Operation` 是按 `op` 字段做的判别联合，`UpdatePlan.operations` 可混合多种 op 顺序写入。

#### 5.5.1 `NodeData`

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `title` | `""` | |
| `description` | `""` | `model_validator` 会在为空时把 description 改成 title，避免空描述。 |
| `importance` | `0.5` | `UPDATE_NODE` 视 0.5 为"未变更"。 |
| `reasoning` | `""` | |
| `ref` | `str \| None` | LLM 用的别名；`apply_add_node` 写完后把 `ref → 真实 node_id` 注册到 `ApplyContext.ref_map` 供后续 op 引用。 |
| `node_id` | `str \| None` | 显式指定 ID（罕见，主要用于回填或重建）。 |
| `grounded_in` | `[]` | 与 op 顶层 `grounded_in` 合并去重。 |
| `metadata` | `{}` | 与运行时注入的 `thread_id` / `source` / `plan_format` 合并。 |

#### 5.5.2 操作类型（`op` 字面量 + 关键字段）

| op | 关键字段 | 拒绝条件 |
| --- | --- | --- |
| `ADD_NODE` | `node_type: NodeType`, `data: NodeData`, `grounded_in`, `reasoning` | description 为空时 apply 阶段跳过；EVENT 允许且每轮 writer 应创建一个。 |
| `ADD_EDGE` | `source_id`, `target_id`, `edge_type=RELATED_TO`, `weight?: float`, `reasoning` | source/target 解析为空时静默跳过。 |
| `UPDATE_NODE` | `node_id`, `data`, `update_reasoning` | 节点不存在 → 跳过。 |
| `REMOVE_NODE` | `node_id` | 节点不存在 → 跳过。 |
| `REMOVE_EDGE` | `source_id`, `target_id`, `edge_type?: EdgeType` | edge_type=None 时按 (src, dst) 删全部匹配边。 |
| `MERGE_NODES` | `node_ids: list[str]` (≥2), `merged_data: MergedNodeData{title, description}`, `reasoning` | 校验器要求 `len(node_ids) ≥ 2`。 |

#### 5.5.3 兼容旧 schema

`UpdatePlan.from_legacy({"nodes":[...], "edges":[...]})`：把旧的 `{nodes, edges}` 平铺结构翻译成 ADD_NODE/ADD_EDGE 操作；EVENT 与其它节点都会转换。如果旧 node 给了 `ref`，会自动追加一条 `current_event -GROUNDS-> ref` 的 ADD_EDGE。`UpdatePlan.parse(raw)` 优先用新 schema，缺失时回退 legacy。

### 5.6 `operations.apply_plan`

- `ApplyContext`：(`store`, `user_id`, `thread_id`, `event_id`, `embed`, `ref_map: dict[str,str]`)；`MemoryService.apply_structured_memory` 总是预填 `ref_map = {"event": event_id, "current_event": event_id}`。
- `ApplyResult`：累计 `nodes / edges / updated / merged / removed_nodes / removed_edges`，`to_summary()` 只暴露前四项给 writer 返回值。
- 调度顺序：**先全部 ADD_NODE，再其它 op**（这样后续 op 才能用刚分配的真实 ID）。
- Handler 对每条 op `try: handler(...) except (KeyError, ValueError) as exc: warn skip` —— 单条失败不会破坏整个 plan。

#### 5.6.1 `_apply_add_node` 关键逻辑

1. description 为空 → skip。
2. `title` 缺省时由 `_title_from_text(description)` 截前 80 字。
3. `grounded_in = [_resolve(ctx, x) for x in op.effective_grounded_in()]`，过滤空串；若全为空则强制等于 `[event_id]`，保证每个新节点至少绑定本 turn 的 event。
4. metadata 强制写入 `thread_id` / `source="memory_writer"` / `plan_format="operations"`。
5. embedding：除 TIME 节点外，调 `embed(f"{title}\n{description}")`。
6. 注册 ref：把 `data.ref` / `data.node_id` / `f"{node_type}_id"` 三个 key（任一非空）→ 真实 ID 写入 `ref_map`。
7. 调 `_add_grounding_edges` 按 `_GROUNDING_RULES` 自动加边：
   - event → concept: `GROUNDS`
   - event → intent: `GROUNDS`
   - event → time: `DEADLINE_FOR`（**source/target 反转**：边方向是 time → event）
   - concept → intent: `TRIGGERS`
   - concept → concept: `RELATED_TO`

#### 5.6.2 `_apply_merge_nodes`

1. 解析 + 取出实际节点列表，少于 2 个直接 skip。
2. **keeper** = 按 `(access_count, importance)` 字典序最大的节点。
3. keeper 描述 = `merged_data.description` 或所有节点 description 去重后用 `\n` 拼接。
4. importance 取所有节点最大值。
5. `grounded_in` 去重并合并。
6. 重新 embed（除 TIME 节点）。
7. metadata 加 `merged: True`。
8. 对其它节点：`replace_edge_endpoint` 把所有指向被合并节点的边改指向 keeper（命中唯一约束的边静默丢弃），删除节点；同时把 `ref_map` 中所有指向被合并 ID 的 ref 改指向 keeper。

### 5.7 `service.MemoryService`

类持有 `store` / `embedder` / `_faiss_indexes: dict[user_id → PersistentFaissIndex]` / `_ranker`。

#### 5.7.1 `MemoryWritePayload`（frozen dataclass）

| 字段 | 含义 |
| --- | --- |
| `user_id`, `thread_id`, `event_id` | 路由键。 |
| `user_message` | 本 turn 用户输入。 |
| `assistant_answer` | 最终 AI 答复（来自 `FinalTextMiddleware`）。 |
| `tool_events` | `state.tool_events` 的快照；writer 只投影完成事件的名称、状态与错误摘要。 |
| `memory_context` | `state.memory_write_context`（由 `MemoryRecallMiddleware` 写入），LLM writer 用来感知"我对哪些已有节点知道什么"。 |

#### 5.7.2 `prepare_turn_contexts` 时序

1. `EventProbe(user_id, text=message, embedding=embed(message))`。
2. 重建 FAISS（保证刚加入的节点已索引）。
3. `entry_matches = vector_index.search(probe.embedding, memory_read_entry_points)`。
4. `read_context = ranker.assemble_read_context(probe, entry_matches, limit=memory_top_k)` —— 喂给 system prompt。
5. `write_context = ranker.assemble_context(probe, limit=memory_top_k)` —— 喂给 memory writer。
6. `replace_recall_snapshot(user_id, _recall_snapshot_items(read_context))` —— 仅写读路径桶到快照表。
7. `touch_nodes(user_id, read+write 节点 IDs)` 提升 `access_count`。
8. `new_id("event")` 只预分配本 turn 的 EVENT ID，不立即写节点或边。
9. 返回 `TurnMemoryContexts(event_id, read_recall, write_recall)`；后续 `MemoryWriter` 的 plan 负责创建真实 EVENT 及其 grounding。

#### 5.7.3 巩固（`consolidate`）

`MemoryWriter.write` 在 `apply_structured_memory` 后自动调一次：

| 步骤 | 行为 |
| --- | --- |
| `merge_similar_concepts` | 两两遍历同 user 的所有 concept，余弦相似度 ≥ 阈值时按 `access_count` 谁高谁活；description 走 `_merge_text`（包含/被包含/拼接）；importance / grounded_in 合并；`replace_edge_endpoint` 改边后删除被合并节点。 |
| `decay_edges` | 全边遍历：`new_weight = weight * exp(-memory_decay_lambda * age_days)`；新权重 < `memory_edge_delete_threshold` → 删除；变化 > 0.001 → `update_edge_weight`。 |
| `complete_orphans` | 找出无入边的 concept，从 events 里挑 top-K 余弦最高、≥ `memory_completion_similarity` 的 event 加 GROUNDS 边，metadata 带 `mechanic="completion"`；只接第一个达标的 event。 |

返回值：`{nodes, edges, updated, merged, decayed, completed}`，被合并到 `MemoryWriter.write` 的最终 dict。

### 5.8 `writer.py`

#### 5.8.1 `MemoryWriter`

- `write(payload)`: `_build_plan(payload)` → `apply_structured_memory(payload, plan)` → `consolidate(payload.user_id)`，结果合并返回。
- `_build_plan`: 仅当 `llm_enabled=True` **且** `model._llm_type != "superassist-fallback"` 才走 LLM；任何异常回退 `_fallback_plan`。
- `_fallback_plan`: 用户消息 < 12 字直接返回空 `UpdatePlan()`；否则用 legacy 形状创建一个 `current_event` EVENT（记录 user/assistant 摘要）和一个 concept `User discussed: ...`。
- LLM prompt 见模块顶部的 `MEMORY_WRITER_PROMPT`，强约束：
  - 输出纯 JSON；仅在存在持久信息时创建 `ref="current_event"` 的 EVENT，纯问候返回空 operations。
  - 边类型默认权重表与 §1.2 一致。
  - 8 条规则，覆盖"只存持久化偏好/目标/事实/概念/截止"、"不存秘密/瞬态工具输出/闲聊"、"相似已有上下文用 UPDATE/MERGE 而非新建"、"中文事件用中文 title/description"。

`_extract_json` 用宽松解析：剥 `\`\`\`json` 围栏 → 取首个 `{` 到末个 `}` 之间内容 → `json.loads`。

`_compact_tool_events` 只保留 `tool_result` 的 `name/status/error_summary≤500`，不发送参数或成功结果。`_compact_memory_node` 只保留 `tier/id/type/title/description/user_id/importance/grounded_in/source/updated_at`，绝不发送 embedding、access_count 或内部 reasoning。

#### 5.8.2 `MemoryWriteQueue`

- 字段：`writer`, `debounce_seconds`, `_queue: deque`, `_lock`, `_timer: threading.Timer | None`。
- `add(payload)`: 加锁入队 → 取消现有 timer → 启动新 daemon timer，到点后调 `flush()`。
- `flush()`: 把 deque 全量取出后释放锁，然后逐条调 `writer.write(payload)`。任何异常被 `logger.exception` 吞掉，**不会再入队重试**（短期内可接受丢一次写入；持久化由 `messages.jsonl` 兜底）。
- CLI 用 `--flush-memory` 在退出前调 `runtime.memory_queue.flush()` 强制写入。

---

<a id="6-agent"></a>
## 6. `agent/`

### 6.1 `agent/state.SuperAssistState`

继承 LangChain `AgentState`（已含 `messages` / `remaining_steps`）：

| 字段 | 类型 | 写入方 | 读取方 |
| --- | --- | --- | --- |
| `user_id` | `str` | `AgentRuntime._initial_state` | 几乎所有 middleware 与工具 |
| `thread_id` | `str` | 同上 | 同上 |
| `input` | `str` | 同上（最新一条用户消息） | `MemoryRecallMiddleware`、`ShortMemoryMiddleware`、`MemoryWriterMiddleware` |
| `memory_event_id` | `NotRequired[str]` | `MemoryRecallMiddleware` | `MemoryWriterMiddleware` |
| `memory_recall` | `NotRequired[dict]` | `MemoryRecallMiddleware` (read 桶) | `DynamicContextMiddleware` |
| `memory_write_context` | `NotRequired[dict]` | `MemoryRecallMiddleware` (write 桶) | `MemoryWriterMiddleware` |
| `rag_mode` | `NotRequired[bool]` | `AgentRuntime._initial_state` | 三条 RAG middleware 与动态提示 |
| `rag_context` | `NotRequired[str]` | `RagRetrievalMiddleware` / `RagRetryMiddleware` | `DynamicContextMiddleware` |
| `rag_sources` | `NotRequired[list[str]]` | 同上 | `RagAttributionMiddleware` |
| `rag_retrieval` | `NotRequired[dict]` | `RagRetrievalMiddleware` | 调试首轮 mode/query/attempt/message |
| `tool_events` | `NotRequired[list[dict]]` | `ToolEventMiddleware`、`SubagentLimitMiddleware` | `MemoryWriterMiddleware`、`ToolCallLimitMiddleware`（计数）|
| `loaded_skills` | `NotRequired[list[str]]` | `ToolEventMiddleware`（探测 read_file 路径）、`ShortMemoryMiddleware`（持久化） | `DynamicContextMiddleware`（注入 SKILL.md 内容）、运行时初始化 |
| `metadata` | `NotRequired[dict]` | 多方更新；运行时初始化时塞入 `history_loaded` / `history_message_count` / `short_memory_summary_loaded` / `loaded_skills` / `tool_calling_enabled` / `tool_schema_binding` | runtime 返回值组装 |

`AgentRunResult.metadata` 主要键：

- `final_assistant_text` — 最终回答（`FinalTextMiddleware` 写入）。
- `messages_path` — 本 thread 的 `messages.jsonl` 绝对路径（`ShortMemoryMiddleware` 写入）。
- `summary` / `summary_updated_at` / `short_memory_compressed` / `short_memory_compressed_records` — 触发了短记忆压缩。
- `model_error` / `model_error_message` — `_error_result` 路径下的失败信息。
- `dynamic_context_injected` — `DynamicContextMiddleware.before_model` 烟测标记。
- `memory_ready` — `FinalTextMiddleware` 标识可以发起持久化写入。
- `rag_trace` — 当轮上传资料检索次数、查询、来源与是否命中证据。
- `answer_provenance` — 最终回答使用的上传文件、网页 URL 与模型知识标记。

### 6.2 `agent/factory.build_agent`

返回 `AgentBundle(agent, settings, model, memory_model, short_memory_model, memory, memory_queue, team_supervisor, team_config_error, rag_service)`。流程：

1. `settings.data_dir.mkdir(...)` 兜底创建目录。
2. `_build_team_supervisor(settings)`：尝试 `AgentTeamConfig.from_file()`，失败即把异常字符串挂在 `team_config_error`。空配置或 `enabled=False` → 返回 `(None, None)`。
3. `set_team_supervisor(team_supervisor)` 写到模块级单例（供 `team_task` 工具与 `AgentRuntime.close` 用）。
4. `create_chat_model(settings)` 创建主 Agent 模型；两次 `create_memory_model` 分别创建 Memory Updater 与短记忆压缩客户端。
5. `MemoryService(...)` + `preload_embedder()`（提前热 BGE）。
6. `MemoryWriteQueue(MemoryWriter(...), debounce_seconds=settings.memory_debounce_seconds)`。
7. `default_tools(...)` 仅在 `enable_tools=True` 时挂载常规工具；`include_team_task` 取决于 supervisor 是否启用。`rag_mode=True` 时额外挂载 `rag_search`、`web_search`、`web_fetch`，网络工具仍检查自己的配置开关。
8. `_build_middleware_chain` 按本文档顶部的固定顺序拼接 middleware；RAG 模式在 recall 后插入 retrieval/retry，在链尾插入 attribution。
9. `compose_system_prompt`（开工具时）或 `SYSTEM_PROMPT` 静态字符串（关工具时）。
10. `create_agent(model, tools, middleware, system_prompt, state_schema=SuperAssistState)`。

`_build_middleware_chain` 对 `subagents_enabled=False` 的情况，不会插入 `SubagentLimitMiddleware`。非 RAG 模式不会构造任何 RAG middleware，保持普通聊天路径不变。

### 6.3 `agent/runtime.AgentRuntime`

- 持有 `_bundle` (AgentBundle)、`rag_mode`、`_run_event_reporter`、`_tool_event_reporter`、`_active_agent_text_seen`（防重复发同一段流式文本）。
- `run(message, *, user_id, thread_id)` 同步路径；`run_streaming(...)` 流式路径，二者都被 `@traceable` 装饰，`process_inputs=without_self`。
- `_initial_state(message, user_id, thread_id)`:
  - thread_id 缺省时生成 `thread_<12hex>`。
  - `_load_thread_metadata` 读 `data_dir/threads/<thread_id>/thread_meta.json`，失败返回 `{}`。
  - `_load_history` 调 `load_short_memory(messages_path, metadata, ...)`，读取摘要检查点后的完整活动段。
  - 初始消息按 `<ShortMemory>` SystemMessage、活动段、最新 `HumanMessage(message)` 排列，并初始化 `rag_mode` / `rag_context` / `rag_sources`。
- `run` 与 `run_streaming` 都用 `rag_turn_context(rag_service, user_id, rag_mode, max_attempts)` 包住整个 Agent 循环，使 `rag_search` 工具和 middleware 共享同一个并发安全的尝试计数器。
- `_stream_agent`: 用 `agent.stream(state, ..., stream_mode=["messages","values"])` 同时收文本块和 state 快照；`accumulate_stream_text` 处理多种 chunk 形态；最终 state 取最后一个 `values` 模式 chunk 或退回初始 state。
- `_report_agent_text`: 用 `_active_agent_text_seen: set[str]` 与 `startswith` 判定避免重复推送同一段渐进文本。
- `_report_tool_event`: 把工具事件透传给 caller 提供的 reporter；同时把 `agent_tool_call` 事件里的 content（即模型在调用工具前的进度说明）当作 `agent_text` 也推一份。
- `_tool_compatibility_metadata`: 关工具时只放 `tool_calling_enabled=False`；开工具且 MiniMax 时额外加 `tool_schema_binding="openai_compatible_minimax"`。
- `_error_result`: 模型抛错时返回中文兜底答复，metadata 记录 `model_error` / `model_error_message` / `final_assistant_text`，避免 channel 看到空答复。

### 6.4 `agent/prompts.py`

- `SYSTEM_PROMPT`：`<role>` / `<thinking_style>` / `<tool_use>` / `<citations>` / `<response_style>` 五块固定 XML-ish 段，写明"被调用工具前先在 assistant content 写一句 NL"等行为约束。
- `subagent_section(max_concurrent)`：限定 `task` 调用并发上限（实际写入文本时 clamp 到 `[1,3]`，与 middleware 一致），列出 `general-purpose` / `research` 两个 subagent 类型。
- `team_section(agents_text)`：把 `TeamSupervisor.available_agents_text()` 列表填入 `<agent_team_system>`。
- `compose_system_prompt(settings, *, team_supervisor, team_config_error)`：`SYSTEM_PROMPT` + 可选 `subagent_section` + 可选 `team_section`；当 supervisor 创建失败 (`team_config_error`)，把错误字符串作为 `Agent team config error: ...` 段拼入，以便 LLM 知道 team 不可用的原因。

### 6.5 `agent/streaming.py`

- `message_text(content)`：兼容 LangChain 的 string / list-of-blocks 两种 content 形态，把所有 string 与 `{type:"text", text}` 块拼回纯文本。
- `merge_stream_text(existing, incoming)`：处理三种增量形态——空、以现有内容前缀为基（直接替换）、现有内容以新增为后缀（保持现有）；其它情况顺序拼接。
- `accumulate_stream_text(buffers, current_message_id, chunk)`：按 message id 归并；返回 `(latest_text, message_id)`，`AgentRuntime` 与 `SubagentExecutor` 共用同一个 buffer 字典。

### 6.6 `agent/short_memory.py`

#### 6.6.1 `ShortMemoryLoad`（frozen dataclass）

| 字段 | 含义 |
| --- | --- |
| `messages: list[BaseMessage]` | 最新摘要检查点之后的完整活动段消息（按时间顺序）。 |
| `records: list[dict]` | 最新摘要检查点之后的原始 JSONL 记录。 |
| `summary: str` | 上一轮压缩输出的 Markdown summary，若无则空串。 |

#### 6.6.2 Token 估算

`estimate_tokens(value)`：使用 `tiktoken` 的 `o200k_base` 编码估算 token；模型输入日志使用同一估算口径。

#### 6.6.3 加载流程 `load_short_memory(messages_path, metadata, ...)`

1. `read_jsonl` 读 messages.jsonl。
2. 从 `metadata.short_memory_compacted_records` 读取最新摘要检查点。
3. 只装载检查点之后的完整活动段，不在加载阶段按回合或 token 静默截尾。
4. `AgentRuntime` 将摘要包装为 `<ShortMemory>` SystemMessage，再依次放入活动段与最新 HumanMessage。

#### 6.6.4 单 turn 写入 `turn_records`

每个完成回合只追加 `{role:user, content, created_at}` 与 `{role:assistant, content, created_at}`。工具参数、原始结果与中间 assistant/tool 消息不会进入后续轮次。

#### 6.6.5 压缩 `maybe_compress_short_memory`

1. 重读检查点之后的活动段；未达到 30 个完成回合且“旧摘要 + 活动段”未达到 80000 tokens 时返回 `{}`。
2. 达到任一阈值后，用 `SUMMARY_SYSTEM_PROMPT` + `build_summary_prompt(previous_summary, active_records, summary_target_tokens, loaded_skills)` 调独立短记忆压缩模型。
3. `messages.jsonl` 不改写；返回新摘要、`summary_version` 与 `short_memory_compacted_records=len(records)` 检查点。
4. 下一轮只装载新检查点之后的记录，从空活动段继续累计，因此正常轮次的模型前缀保持追加式增长。
5. LLM 失败 → 返回 `{short_memory_compression_error: "..."}`。

`SUMMARY_SYSTEM_PROMPT` 强调要保留偏好 / 当前任务 / 工具事实 / loaded skills，丢弃长 webpage 内容与重复问候。

---

<a id="7-middlewares"></a>
## 7. `middlewares/`

LangChain 1.x middleware 的钩子分类：`before_agent` / `after_agent`（agent 起止）、`before_model` / `after_model`（每次模型调用前后）、`wrap_model_call`（包裹一次模型调用）、`wrap_tool_call`（包裹一次工具调用）。**注册顺序决定 before_* 顺序，after_* 反向**。本节列出每条 middleware 的关键合同与状态读写。

### 7.1 `ToolErrorMiddleware`

- 钩子：`wrap_tool_call`。
- 行为：`try handler(request) except Exception as exc: return ToolMessage(content=f"{tool_name} failed: {exc}", status="error")`。
- 必须放在 `wrap_tool_call` 链最外层，让其它中间件不需要再处理异常。

### 7.2 `ToolCallLimitMiddleware`

- 状态读：`state.tool_events`（计数 `tool_result` 个数作为已用预算）。
- 行为：达到上限直接返回错误 `ToolMessage`；上限 `0` 等于禁用全部工具调用。
- 注意：因为 `ToolEventMiddleware` 在本中间件之后才把当前 result append 进 `tool_events`，**当前调用本身不计入预算**——预算是"在此之前已完成的工具数量"。

### 7.3 `MemoryRecallMiddleware`

- 钩子：`before_agent`（一次性，整个 invoke 周期只跑一次）。
- 短路：`state.memory_event_id` 已存在 → 跳过（重入保护）。
- 调 `MemoryService.prepare_turn_contexts(user_id, thread_id, message)`。
- 写入 state：`memory_event_id` / `memory_recall` / `memory_write_context`。主模型 recall 只保留 `tier/id/type/title/description/user_id/updated_at`；writer 额外保留 `importance/grounded_in/source`。两者都排除 embedding、access_count 和内部 reasoning。

### 7.4 `DynamicContextMiddleware`

- 钩子：`before_model`（写入 `metadata["dynamic_context_injected"]=True` 用作 trace 烟测）+ `wrap_model_call`（核心）。
- `wrap_model_call`：
  1. 从 state 抽 `memory_recall` / `loaded_skills` / `user_id` / `thread_id`；RAG 模式同时读取 `rag_context` 与本轮 session trace。
  2. 可用 skill 索引由 `compose_system_prompt` 放入稳定静态前缀；动态层只用 `build_loaded_skills_section` 注入本轮开始时仍处于激活期的 skill 全文。
  3. 拼装 `<TurnContext>`，内部按 `<RuntimeContext>`、`<LongTermMemory>`、可选 `<ActiveSkills>` / `<RAGContext>` 分区。
  4. RAG 模式加入封闭证据规则：上传文件是不可信资料，不得虚构引文；证据不足应继续 `rag_search`；耗尽上传检索后才按联网开关降级，并区分资料/网页/模型知识。
  5. GPT-5.6 路径把 TurnContext 插到最新 HumanMessage 前，因此初次调用始终以当前用户消息结尾；工具续调用中，AI/tool 消息仍自然位于当前用户之后。

### 7.5 `ShortMemoryMiddleware`

- 钩子：`after_agent`（**注册顺序在 DynamicContextMiddleware 之后，因此 after_agent 反向时它在 MemoryRecallMiddleware 之后、ToolEventMiddleware 之前**——但 after_agent 与 after_model 钩子不冲突）。
- 流程见 §6.6；返回 `{"metadata": metadata, "loaded_skills": loaded_skills}` 让 LangChain 把 metadata 与 loaded skills 合并进 state（`loaded_skills` 排序去重后写回，确保跨 turn 稳定）。
- 持久化文件：`<data_dir>/threads/<thread_id>/messages.jsonl` 与 `thread_meta.json`。元数据持久化 `user_id`（Go 会话所有权与管理员审计依据）、`loaded_skills`、`summary`、`summary_updated_at`。

### 7.6 `ToolEventMiddleware`

- 钩子：`wrap_model_call`（仅记录 `agent_tool_call` 事件 —— `AIMessage.tool_calls` 非空时）+ `wrap_tool_call`（记录 `tool_start` / `tool_result`）。
- `wrap_tool_call` 关键副作用：
  1. 把 `tool_start` event append 到 state.tool_events 并 reporter 上报。
  2. 调底层 handler 拿 `result: ToolMessage`。
  3. 当 tool_name == `read_file` 时，调 `skill_name_from_virtual_path(args["path"])`：若该路径形如 `/mnt/skills/public/<name>/SKILL.md`，把 `<name>` 加入 `state.loaded_skills`，不重复添加。
  4. 构造 `tool_result` event：`{type, tool, args, content, status}`，错误时附 `error=content`，loaded_skills 非空时附在事件 metadata。
- `_message_text` / `_agent_tool_call_event` 与 `streaming.message_text` 一致地处理 list-of-blocks 内容。

### 7.7 `SubagentLimitMiddleware`

- 钩子：`after_model`，仅当本 turn 启用了 subagents 才会注入。
- 行为：扫描最后一条 `AIMessage.tool_calls`，找出 `name == "task"` 的索引；若超过 `max_concurrent` → 仅保留前 `max_concurrent` 个，**其它直接从 tool_calls 中剔除**（不会触发 ToolNode 执行），并在 state.tool_events 追加一条 `{type: subagent_limit, max_concurrent, dropped}` 事件。
- `max_concurrent` 在构造时就 clamp 到 `[1,3]`。

### 7.8 `MemoryWriterMiddleware`

- 钩子：`after_agent`（注册在 `ShortMemoryMiddleware` 之后，但 after_agent 反向，因此本 middleware 实际**先**于 ShortMemoryMiddleware 跑——并不重要，因为它只入队）。
- 短路：`state.memory_event_id` 缺失则 skip（保护测试和外部入口直接 invoke agent 的场景）。
- 调 `MemoryWriteQueue.add(MemoryWritePayload(...))`，**只入队**；真正写库由 timer 后台触发或 CLI `--flush-memory` 强制 flush。
- payload.assistant_answer 取最后一条 `AIMessage` 文本；若仍为空，writer 的 fallback 逻辑会因 user_message 太短而生成空 plan，最终只跑一次 `consolidate`。

### 7.9 `FinalTextMiddleware`

- 钩子：`after_agent`。
- 行为：从 `state.messages` 末尾找第一条非空 `AIMessage`，将其文本写到 `metadata.final_assistant_text` 并设 `metadata.memory_ready=True`。`AgentRuntime._result_from` 优先读 `metadata.final_assistant_text` 作为 `AgentRunResult.answer`。

### 7.10 `RagRetrievalMiddleware`

- 仅在 `rag_mode=True` 时注册，钩子为 `before_agent`，注册在 `MemoryRecallMiddleware` 之后，因此每轮只执行一次首检。
- 使用原始 `state.input` 执行 `mix` 检索；没有活动 session 时写入明确的 unavailable 状态。
- 成功时把结构化实体、关系和原文证据写入 `rag_context`，来源文件写入 `rag_sources`，并把首轮结果摘要写入 `rag_retrieval`；失败不会抛出到整轮聊天，而是保留 session trace 给重试链。

### 7.11 `RagRetryMiddleware`

- 仅在 RAG 模式注册，钩子为 `after_model`。
- 当模型准备直接结束、当前 session 尚无成功证据且未耗尽次数时，把最后一条 AIMessage 改为强制 `rag_search` 工具调用。
- attempt 2 使用 `naive` 和面向关键术语/直接证据的确定性改写；attempt 3 使用 `global` 和面向实体/别名/关系的改写。模型主动调用 `rag_search` 也消耗同一额度。
- 达到 `SUPERASSIST_RAG_MAX_ATTEMPTS` 后不再注入调用，允许模型按动态提示执行联网或保守知识降级。

### 7.12 `RagAttributionMiddleware`

- 仅在 RAG 模式注册，钩子为 `after_agent`；因注册在链尾，反向执行时先生成确定性的来源尾注，再由其它 after-agent middleware 持久化最终文本。
- 汇总 session 中成功的上传文件和 `web_search` / `web_fetch` 工具结果中的 URL，过滤错误输出。
- 在最终回答末尾追加“回答依据”，并写入 `metadata.rag_trace` 与 `metadata.answer_provenance`；没有上传或联网证据时显式标记模型自身知识及上传检索次数。

---

<a id="8-subagents"></a>
## 8. `subagents/`

子智能体是 **进程内** 的一次性 agent，通过 `task` LangChain 工具触发；与 `team_task` 启动外部 ACP 进程是两条完全独立的链路。

### 8.1 `subagents/config.py`

`SubagentConfig`（frozen dataclass）：

| 字段 | 含义 |
| --- | --- |
| `name` | `general-purpose` / `research`，作为 `task(subagent_type=...)` 的取值。 |
| `description` | 文案，用在 `subagent_section` system prompt 中。 |
| `system_prompt` | 子 agent 的系统提示（见 `GENERAL_PURPOSE_PROMPT` / `RESEARCH_PROMPT`）。 |
| `allowed_tools` | `None` 表示继承除 `task` 外所有 lead 工具；`research` 限制为 `[web_search, web_fetch, read_file, list_files, write_file]`。 |
| `timeout_seconds` / `max_turns` | 来自 settings。 |

`build_builtin_subagents(timeout_seconds, max_turns)` 返回 `dict[name → SubagentConfig]`。

`GENERAL_PURPOSE_PROMPT` / `RESEARCH_PROMPT` 都强制不得递归调用 `task` 工具，并要求结构化输出（含 citations）。

### 8.2 `subagents/registry.py`

`SubagentRegistry(settings)`：构造时一次性 build 所有 builtin configs。`get_subagent_config(name, settings)` / `get_available_subagent_names(settings)` 是模块级便利函数，每次都新建 registry —— 因为内部只是 dict 构造，没有副作用。

### 8.3 `subagents/store.py`

#### 8.3.1 `SubagentStatus`（StrEnum）

`pending` → `running` → `completed` / `failed` / `timed_out`。

#### 8.3.2 `SubagentResult`（dataclass）

| 字段 | 含义 |
| --- | --- |
| `task_id` | `subagent_<12hex>`。 |
| `description` | 调用方传入的简短描述。 |
| `subagent_type` | 与 `SubagentConfig.name` 对齐。 |
| `status` | 见上。 |
| `result` | 最终 AI 答复字符串（completed 才有意义）。 |
| `error` | 失败/超时原因。 |
| `ai_messages` | 流程中所有 AIMessage 文本（按顺序），UI 用来展示思考过程。 |
| `started_at` / `completed_at` | UTC。 |

`to_dict()` 显式把 enum 转字符串、datetime 转 ISO，给 FastAPI 返回。

#### 8.3.3 `SubagentTaskStore`

进程内 LRU `dict` + `deque`，最多 `max_items=200`；`put` 在已存在时只覆盖不刷顺序——**newest 永远在 deque 头部**（`appendleft`）。`list(limit)` 取 `deque[:limit]`。模块底部暴露单例 `TASK_STORE` 给执行器与 UI 共享。

### 8.4 `subagents/executor.SubagentExecutor`

#### 8.4.1 构造

接收 `SubagentConfig`、`tools`、`settings`、`run_event_reporter`。`tools` 经 `_filter_tools` 处理：永远剥掉 `task`（防递归），再按 `config.allowed_tools` 白名单过滤。`model = create_chat_model(settings)` —— 与 lead agent 共享同一模型工厂。

#### 8.4.2 `LangGraph` graph

子 agent **的确**包了一层 LangGraph：`StateGraph(dict)` 三节点 prepare → agent → finalize。

- `_prepare`：把 prompt 和 system prompt 拼成初始 messages。
- `_agent`：内部 `create_agent(model, tools)`（**没有** middleware 链，纯净的 LangChain agent）；调 `_invoke_agent` 走 stream 或 invoke，捕 `GraphRecursionError` → `_summarize_after_recursion_limit` 强制让模型生成"达到 max_turns"的总结返回，避免子任务静默挂起。
- `_finalize`：`_last_ai_text` 取尾部 `AIMessage`，写回 `holder.result` / `status=COMPLETED` / `completed_at`。

`recursion_limit` 通过 `config.max_turns` 注入；其它 trace metadata（task_id / description / subagent_type）随 `runnable_trace_config` 上行 LangSmith。

#### 8.4.3 同步 / 异步桥接

- `arun(prompt, *, result)`：`asyncio.timeout(config.timeout_seconds)` 包裹 `asyncio.to_thread(self.graph.invoke, ...)`。
- `run(prompt, *, task_id, description)`：`_run_coro_sync` —— 已有事件循环则用 `asyncio.Runner` 起 nested loop，否则 `asyncio.run`。
- 任何异常 → `holder.status = FAILED`，error = `f"{ExcType}: {msg}"`。

#### 8.4.4 流式去重

`_reported_subagent_text_seen: set[str]` 配合 `startswith` 判定，避免把"已经报过的前缀+新增内容"再推一次给 reporter；reporter 接收的事件 type 固定为 `subagent_text`。

### 8.5 `tools/task.py`（lead 侧入口）

- `_semaphore = BoundedSemaphore(value=3)` —— 进程级硬上限，与 `subagent_max_concurrent` 默认值同步。
- `task(description, prompt, subagent_type="general-purpose")`：subagents 关闭返回错误串；type 不存在时返回 `available` 列表；信号量 acquire 走 `timeout = config.timeout_seconds`，超时返回 `"Task timed out. Error: No subagent slot available after Xs"`。
- 命中槽位后构造 `SubagentExecutor`：`tools=default_tools(include_task=False)`（再次保险禁用嵌套 task），`run_event_reporter` 优先用绑定时传入的，其次 fallback 到 `current_run_event_reporter()`。
- 返回字符串格式：`Task Succeeded. Result: ...` / `Task timed out. Error: ...` / `Task failed. Error: ...`，由 lead agent 自然地拼回回答。

`make_task_tool(reporter)` 是带闭包的工厂版本——`AgentRuntime` 在拿到 `tool_event_reporter` 后用它替换全局 `task` 实例，让流式 reporter 能贯穿父子层。

---

<a id="9-acp_client"></a>
## 9. `acp_client/`

唯一直接 `import acp` 的子包；其它代码都通过 `ACPSession` 接口与 ACP 通信。

### 9.1 `acp_client/loop.AsyncLoopThread`

每个 `TeamMember` 持有一个 `AsyncLoopThread` 实例。`__init__` 用 `asyncio.new_event_loop()` 在 daemon thread 中调 `run_forever()`；`submit(coro)` 把协程派发到该 loop，返回 `concurrent.futures.Future`，调用方在主线程 `.result()` 等待。`close()` 用 `call_soon_threadsafe(loop.stop)` 优雅关停，最多 join 5 秒。

LangChain 的同步执行模型不允许阻塞主线程跑 ACP 异步协议，这是必须的桥接层。

### 9.2 `acp_client/permissions.PermissionPolicy`

| 值 | 行为 |
| --- | --- |
| `AUTO_APPROVE` | 找首个 `kind == allow_once` 或 `allow_always` 的 option，构造 `RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", optionId=...))`。 |
| `DENY` | 直接返回 `DeniedOutcome(outcome="cancelled")`。 |

匹配逻辑通过 `option.option_id` 或 camelCase `optionId` 兼容不同 ACP 实现。

### 9.3 `acp_client/process`

#### 9.3.1 `ACPSpawnRequest`（dataclass）

| 字段 | 含义 |
| --- | --- |
| `name` | Agent 名（用于错误信息与日志）。 |
| `command` / `args` | 启动命令；`shutil.which(command)` 解析后再 spawn。 |
| `cwd` | 工作目录（每个 thread 一份，由 `TeamLedger.workspace_dir(thread_id, agent)` 确定）。 |
| `env` | 经 `resolve_env` 展开 `$VAR` 为 `os.environ` 值，并补 `npm_config_cache` / `prefer_offline=true` / `audit=false` / `fund=false` 提升 npm 启动速度。 |
| `model` | 可选；非空时作为 `new_session(model=...)` 参数透传。 |
| `permission_policy` | 默认 `AUTO_APPROVE`，由 `agent_team.toml` 的 `auto_approve_permissions` 翻译而来。 |

#### 9.3.2 `ACPSession`（dataclass）

字段：`name` / `context`（`spawn_agent_process` 返回的 async context manager）/ `conn`（ACP `Connection`）/ `session_id` / `client`（自定义 `_CollectingClient`）。

- `prompt(text)`：把 `client.chunks` 长度作为基线，发送 `text_block(text)` 的 prompt，等待返回，把基线之后追加的 chunks `"".join(...)` 作为整段响应；空字符串退化为 `"(no response)"`。
- `close()`：调 `context.__aexit__(None, None, None)`，异常被 `logger.exception` 吞掉以避免影响其它 session。

#### 9.3.3 `open_session(request)`

1. `import acp` 失败 → `ACPClientError`（缺少依赖时给出明确指引）。
2. 解析 cwd 并 mkdir。
3. `_CollectingClient`：
   - `session_update`：当 update 含 `TextContentBlock` 时把文本片段 append 到 `chunks`。
   - `request_permission`：调 `build_permission_response(options, policy)`，并在 logger 上记录 auto-approve / deny。
4. `spawn_agent_process(client, command, *args, env=..., cwd=...)` → `await context.__aenter__()` 拿 `(conn, _proc)`。
5. `conn.initialize(protocol_version, ClientCapabilities(), Implementation(name="superassist", title="SuperAssist", version="0.1.0"))`。
6. `conn.new_session(cwd=str(cwd), mcp_servers=[], model=request.model?)`，得到 `session_id`。
7. 失败映射：`FileNotFoundError` → `missing_command_message`（特别处理 `codex-acp` 不存在但 `codex` 存在的常见情况，提示用 `npx -y @zed-industries/codex-acp`）；其它 → `format_start_error`（识别 `EPERM` + `npm-cache` 的常见 Windows npm 缓存权限问题给出明确建议）。

#### 9.3.4 `resolve_env(env, *, cache_dir)`

- `value.startswith("$")` 时按 `os.environ.get(value[1:], "")` 展开。
- `cache_dir` 非 None 时强制写入 `npm_config_cache=<abs path>` + 三个 npm 性能开关（仅当 caller 没有显式设置时通过 `setdefault`）。
- 调用方：`teams.supervisor.TeamMember._ensure_session` 用 `workspace.parents[3] / "npm-cache"` 作为 cache_dir，对应到 `<data_dir>/teams/default/npm-cache`。

### 9.4 错误格式 `acp_client/errors`

- `ACPClientError`：`RuntimeError` 子类，全包内 ACP 异常的统一类型。
- `missing_command_message(name, command)`：识别 `codex-acp` 的常见误装情况；通用情况返回安装建议。
- `format_start_error(name, exc)`：把 `code` / `data` 属性塞进消息；空 / `Internal error` 提示去手动跑命令看 stderr；`EPERM` + `npm-cache` 提示走项目本地 cache。

---

<a id="10-teams"></a>
## 10. `teams/`

### 10.1 `teams/config.py`

#### 10.1.1 `TeamAgentConfig`

| 字段 | 默认 | 校验 |
| --- | --- | --- |
| `name` / `command` / `description` | — | 字段校验器 `_non_empty` strip 后非空。 |
| `args` | `[]` | |
| `env` | `{}` | 值支持 `$NAME` 占位符，由 `resolve_env` 展开。 |
| `model` | `None` | 透传给 `new_session(model=...)`。 |
| `auto_approve_permissions` | `False` | True → `PermissionPolicy.AUTO_APPROVE`，否则 `DENY`。 |

#### 10.1.2 `AgentTeamConfig`

| 字段 | 默认 | 备注 |
| --- | --- | --- |
| `enabled` | `False` | 顶层开关。 |
| `idle_ttl_seconds` | `3600` | 后续 `sweep_idle` 用；`<=0` 等于禁用空闲清理。 |
| `agents` | `[]` | model_validator 拒绝重名。 |

`from_file(path=None)`：默认 `<PROJECT_ROOT>/agent_team.toml`，文件不存在返回 `disabled()`；解析/校验失败抛 `AgentTeamConfigError`（`ValueError` 子类），上层捕到后塞到 `bundle.team_config_error`，**不**拒绝服务。

### 10.2 `teams/ledger.TeamLedger`

每个 thread 的 ledger 路径：`<root>/threads/<safe_name>/ledger.jsonl`。配套目录：`inbox/<agent>.jsonl` / `outbox/<agent>.raw.jsonl` / `workspaces/<agent>/`。

#### 10.2.1 `append_message` 记录字段

| 字段 | 含义 |
| --- | --- |
| `id` | `msg_<32hex>`。 |
| `seq` | 1-based 单调递增；与文件中位置一致。 |
| `thread_id` | 当前 thread。 |
| `sender` / `recipient` | 例：`superassist` ↔ `claude_code`。 |
| `kind` | `task` / `result`，可扩展。 |
| `body` | 原始 prompt 或 result 文本。 |
| `artifact_paths` | 关联文件路径数组（v1 暂未用）。 |
| `parent_ids` | result 记录指向 task 记录 id；构成因果链。 |
| `created_at` | UTC ISO。 |
| `prev_hash` | 上一条 `hash` 或 `GENESIS_HASH = "0"*64`。 |
| `extra` | 任意 dict（描述、附加元信息）。 |
| `hash` | `sha256(canonical_json(record without {"hash","sig"}))`，hex。 |
| `sig` | `hmac_sha256(supervisor.key, canonical_json(record without {"sig"}))`，hex。 |

`_canonical_bytes`: `json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")` —— 严格保证 hash/sig 可重现。

`_validate_records` 在每次 `append_message` 之前与每次 `read_ledger` 时被调用：检查 seq 连续、prev_hash 链匹配、hash 自洽、`hmac.compare_digest(sig, expected)`。任何失败抛 `LedgerTamperError`（`LedgerError` 子类，进而是 `RuntimeError`）。

#### 10.2.2 文件锁 + 密钥

- `FileLock(str(path) + ".lock")`：append_message / read_ledger / append_inbox / append_raw 与密钥读取都以独立 lock 文件保护。
- `supervisor.key`：32 字节随机串（`secrets.token_bytes`），首次访问时生成；hex 编码持久化。生产部署里建议把这个密钥与数据目录一起备份；丢失等于无法验证 ledger。

#### 10.2.3 `_safe_name(value)`

把非字母数字与 `-_.` 之外的字符替换为 `_`，最终去除首尾 `._`，空串 → `default`。同样规则用于 thread_id 与 agent 名，防止路径穿越。

### 10.3 `teams/supervisor`

#### 10.3.1 `TeamSupervisor`

字段：`config: AgentTeamConfig`、`settings`、`bus: TeamLedger(default <data_dir>/teams/default)`、`_member_factory: type[TeamMember]`、`_members: dict[name → TeamMember]`、`_lock: threading.Lock`。

- `enabled` 属性：`config.enabled and bool(config.agents)`。
- `available_agents_text()`：拼成 `"- name: description"` 多行串供 system prompt 使用。
- `invoke(agent, *, thread_id, description, prompt, wait=True)`：
  1. 校验 enabled / wait==True / agent 在配置中。
  2. `bus.validate_thread(thread_id)` 提前发现 ledger 被篡改 —— **直接抛 `LedgerTamperError`**（不被翻译为 `TeamSupervisorError`），让 caller 区别对待。
  3. 写一条 `kind=task` 主链记录 + 一条 inbox 记录。
  4. `bus.workspace_dir(thread_id, agent)` 准备工作目录（每个 (thread, agent) 一份）。
  5. 调 `_member(config).invoke(thread_id, prompt, workspace)` 拿响应字符串；ACPClientError → 翻 `TeamSupervisorError`。
  6. 写一条 outbox 原始日志 + 一条 `kind=result` 主链记录（`parent_ids=[task.id]`）。
  7. 返回 `TeamTaskResult(agent, task_id, result, ledger_id)`。
- `sweep_idle()` / `close()`：暂未在 lead 周期主动调用，但会在 `AgentRuntime.close()` 中显式 close 当前 supervisor 的所有 member。

#### 10.3.2 `TeamMember`

| 字段 | 含义 |
| --- | --- |
| `config: TeamAgentConfig` | |
| `last_used: float` | `time.monotonic()`，`sweep_idle` 比对用。 |
| `_loop: AsyncLoopThread` | 名为 `team-agent-<name>`。 |
| `_sessions: dict[(thread_id, workspace_str) → ACPSession]` | 同一对组合复用 session，跨 turn 保持 ACP 上下文。 |
| `_closed: bool` | |

- `invoke(thread_id, prompt, workspace)`：在 loop 上 `submit(_aprompt(...))` 并 `.result()`（同步阻塞当前线程，但不阻塞 lead agent 自己的事件循环——LangChain 已经是同步上下文）。
- `_ensure_session(thread_id, workspace)`：缓存命中返回；否则构造 `ACPSpawnRequest`（注入 `npm_config_cache=<workspace.parents[3]/npm-cache>`，policy 由 `auto_approve_permissions` 翻译），`open_session(request)` 后落 dict。
- `close()`：在 loop 中关闭所有 session 后 close loop，整个流程 `result(timeout=10)` 兜底。

#### 10.3.3 `team_thread_context`

`teams/context.py` 暴露 `ContextVar` 与 `team_thread_context(thread_id)` 上下文管理器。`AgentRuntime.run*` 用它包裹 agent 调用，让 `tool team_task` 在工具执行栈深处仍能拿到当前 thread id（不必把 thread_id 透传到工具签名）。

### 10.4 `tools/team.team_task`

LangChain `@tool("team_task")`：

| 入参 | 校验 |
| --- | --- |
| `agent` | 必填；不在配置内 → 在 supervisor 抛错 → 工具返回 `Error: Unknown team agent ...`。 |
| `description` | 简短任务描述。 |
| `prompt` | 主指令。 |
| `wait` | v1 必须 `True`，否则抛 `TeamSupervisorError`。 |

工具实现把所有错误翻译成可读字符串：`Error: ...` 让 LLM 自行恢复，不会让 LangChain 感知到异常（因为 `ToolErrorMiddleware` 还会兜底）。

---

<a id="11-tools"></a>
## 11. `tools/`

`default_tools(include_task=True, include_team_task=False, run_event_reporter=None)` 返回工具列表：`[echo, list_files, read_file, write_file, delete_path, web_search, web_fetch, shell, (task), (team_task)]`。`task` 默认替换为 `make_task_tool(reporter)` 绑定版本，使流式 reporter 可下钻到子 agent。

### 11.1 `tools/basic.py`

| 工具 | 行为 |
| --- | --- |
| `echo(text) -> str` | 直接返回 `text`，烟测用。 |
| `current_time() -> str` | 返回 `datetime.now(UTC).isoformat()`，**未挂入 `default_tools`**（运行时由 DynamicContextMiddleware 把 current time 注入 system prompt，不需要工具）。 |

### 11.2 `tools/files.py` —— 工作区沙箱

所有路径都被 `_resolve_workspace_path` clamp 到 `Settings.resolved_tool_workspace_dir.resolve()`：

- 解析后调 `resolved.relative_to(root)`，越界抛 `PermissionError("Path is outside the tool workspace: ...")`。
- 不存在的目录会被 mkdir 创建为工作区根。

`read_file` 还接受 `<skill virtual path>`（见 §12），命中时跳出工作区限制读 `skills/` 目录。

| 工具 | 关键参数 | 行为 |
| --- | --- | --- |
| `list_files(path=".", max_depth=2)` | depth clamp 到 `[0,8]` | rglob 全表，过滤层级；每行 `<rel>/`（dir）或 `<rel>`（file），最多 300 行后追加 `... [truncated]`。 |
| `read_file(path, start_line?, end_line?)` | start/end 都给才裁切 | UTF-8 + `errors="replace"`；最终内容 clamp 至 50000 字符。 |
| `write_file(path, content, append=False)` | mode `"a"` / `"w"` | 自动 mkdir 父目录；返回 `OK`。 |
| `delete_path(path, recursive=False)` | 只删工作区内的文件或目录 | `recursive=True` 调 `shutil.rmtree`；目录非空且 `recursive=False` 时 `rmdir` 自身抛错由外层 `ToolErrorMiddleware` 包成 ToolMessage。 |

### 11.3 `tools/web.py`

| 工具 | 行为 |
| --- | --- |
| `web_search(query, max_results=5)` | clamp 到 `[1,10]`；先试 `lite.duckduckgo.com/lite`，失败再试 `duckduckgo.com/html`。返回 `json.dumps([{title, url, snippet}], indent=2)`。`_normalize_result_url` 还会解 DuckDuckGo `/l/?uddg=` 重定向。 |
| `web_fetch(url, max_chars=12000)` | clamp 到 `[1000, 50000]`；UA 写明 `SuperAssist/0.1`；非 plain-text 用 `_clean_html`（剥 `<script>`/`<style>` 后再 untag）。 |

`_ensure_network_enabled()` 控制 `tool_network_enabled` 开关，闭合时所有联网工具都立即返回 `Error: Network tools are disabled by SUPERASSIST_TOOL_NETWORK_ENABLED=false`。

`_fetch_url`：1MB 读上限；从 `Content-Type` 抓 charset，缺失默认 utf-8；`errors="replace"` 容错。

### 11.4 `tools/shell.py`

工具关闭即返回错误串（`SUPERASSIST_TOOL_SHELL_ENABLED=false`）。打开后：

- `_DANGEROUS_PATTERNS`：拒绝 `Remove-Item ... -Recurse -Force`、`rm -rf`、`git reset --hard`、`git checkout --`、`del /s|q`、`rmdir /s`、`format X:`，命中即返回 `Error: shell command blocked: destructive command requires manual execution`。
- `_resolve_cwd(cwd)`：相对路径以 `PROJECT_ROOT` 为根；`relative_to` 越界抛 `PermissionError`；不存在或非目录也抛错。
- `_get_shell()`：Windows 优先 `pwsh` → `powershell` → 系统 PowerShell → `cmd.exe`；POSIX 优先 `zsh` → `bash` → `sh`。
- `_shell_args`：PowerShell 使用 `-NoProfile -ExecutionPolicy Bypass -Command`；cmd 用 `/c`；POSIX 用 `-c`。
- timeout = `clamp(settings.tool_shell_timeout_seconds, [1, 600])`；输出包含 stdout / stderr / 非零 ExitCode 文本，最终 `_truncate(output, max_chars=tool_shell_output_max_chars)`，超长时中段替换为 `... [truncated N chars] ...`。

### 11.5 `tools/task.py`

见 §8.5。

### 11.6 `tools/team.py`

见 §10.4。

### 11.7 `rag/tools.py`

`rag_search(query, mode="mix")` 从当前 `ContextVar` 获取 `RagTurnSession`，不接受调用方传入 `user_id`，因此模型无法越权检索其他用户目录。模式白名单最终由 `LightRAGService.retrieve` 校验为 `mix|hybrid|local|global|naive`，非法值回退到 `mix`。成功返回 `RAG_RETRIEVAL_SUCCESS + Sources + context`，失败返回 `RAG_RETRIEVAL_FAILED + attempts`，不抛出工具异常。

---

<a id="12-skills"></a>
## 12. `skills/`

#### 12.1 路径与 SKILL.md

- 物理目录：`<PROJECT_ROOT>/skills/`，公开 skill 在 `skills/public/<name>/SKILL.md`。
- 虚拟容器路径：`/mnt/skills/`（统一供 LLM 输出，便于跨平台展示，不依赖运行机器实际路径）。
- 单个 skill 的 `virtual_file_path`：`/mnt/skills/public/<name>/SKILL.md`。

#### 12.2 `Skill`（frozen dataclass）

| 字段 | 含义 |
| --- | --- |
| `name` | frontmatter `name:` 或目录名兜底。 |
| `description` | frontmatter `description:`（可空）。 |
| `skill_dir` / `skill_file` | 物理路径。 |
| `relative_path` | 相对 `skills/` 的相对路径（用于拼虚拟路径）。 |

`_parse_skill_file` 用 `_parse_frontmatter` 解析 `---` 包裹的 YAML-ish frontmatter（仅简单 `key: value` 行，剥首尾引号），不依赖 `pyyaml`。

#### 12.3 注入 system prompt

- `build_available_skills_section()`：列出所有 public skill 的 `<skill_system>` XML 块，列出 name / description / location（虚拟路径）。LLM 看到匹配场景时按指引调 `read_file` 读虚拟路径加载 skill 内容。
- `build_loaded_skills_section(loaded_skills)`：对每个已加载 skill 读取 SKILL.md 全文，包成 `<skill name="...">...</skill>` 段——这就是"读过 SKILL.md → 后续 turn 也保留 skill 上下文"的实现机制。
- `_list_public_skills_cached`：`lru_cache(1)`，进程生命周期只扫一次。

#### 12.4 路径解析

- `resolve_skill_virtual_path(path)`：以 `/mnt/skills/` 开头时，解析为 `<PROJECT_ROOT>/skills/<rest>`，再 `relative_to(SKILLS_ROOT)` 防越界，命中则返回真实 `Path`，否则返回 `None`。`tools/files.py:read_file` 优先调它，再退回工作区路径。
- `skill_name_from_virtual_path(path)`：仅识别 `/mnt/skills/public/<name>/SKILL.md`，返回 `<name>`，用于 `ToolEventMiddleware` 探测"agent 刚加载了哪个 skill"。

---

<a id="13-channelsfeishu"></a>
## 13. `channels/feishu`

### 13.1 `FeishuInboundMessage`（dataclass）

| 字段 | 含义 |
| --- | --- |
| `chat_id` / `message_id` / `sender_open_id` | Lark 标识。 |
| `text` | strip 后的纯文本。 |
| `root_id` | 群里"回复"消息的根 ID；私聊为 None。 |
| `chat_type` | `p2p` / `group` 等。 |
| `mentions` | dict 列表，至少含 `name` / `open_id`。 |
| `files` | `[{file_key} 或 {image_key}]`；图片进入多模态主 Agent，其他文件仍返回 unsupported。 |
| `topic_id` 属性 | `root_id or message_id`，用作 `FeishuThreadStore` 的 topic 维度。 |
| `is_private` 属性 | `chat_type in {"p2p","private","single"}`。 |

### 13.2 `FeishuChannel`

#### 13.2.1 启动

`start()`：
1. 缺 app_id/secret → 抛 `RuntimeError`。
2. 懒导入 `lark_oapi`（缺包错误信息明确）。
3. 缓存所有 lark 请求/响应类型到实例字段（避免在每个发送路径里再做 import）。
4. 用 `lark.Client.builder().app_id(...).app_secret(...).domain(...).build()` 建 API client。
5. `await asyncio.to_thread(get_embedder(self.settings).preload)` —— 在 channel 主线程外预热 BGE。
6. 启动 daemon 线程跑 `_run_ws`（每个事件 → `asyncio.run_coroutine_threadsafe(handle_inbound(inbound), main_loop)`）。

#### 13.2.2 处理入站

`handle_inbound(inbound)`：
1. 白名单：`feishu_allowed_open_id_set` 非空时严格匹配 sender。
2. `should_trigger_agent`：私聊立即触发；群里需 `mention_only=False` 或确实有 @ 提及。
3. `clean_mention_text` 去掉 @ 文本。
4. scope 正忙时直接忽略新消息；图片以外的附件返回 `UNSUPPORTED_FILE_MESSAGE`。
5. 每张图片按 10 MB 下载上限、每条消息最多 4 张；PNG/JPEG/GIF/WebP 原始字节直接进入模型，不校正方向、不缩放、不铺白、不转码。
6. 可选 RapidOCR 在本地线程中逐图识别，结果标为不可信辅助文本；OCR 缺包、初始化或识别失败时 fail-open。
7. 原图 Base64、OCR、当前用户问题组成一个 LangChain 多模态 `HumanMessage`，直接进入主 Agent。最新一组原图按 180 秒滑动 TTL 跨飞书消息保留，每条有效后续消息刷新计时；过期后清除原始字节，只复用历史中的 OCR、回答和 `<ImageDescription>`。同一轮 Skill/工具循环的每次模型调用也完整保留原图；GPT-5.6 显式关闭 `use_previous_response_id`，避免客户端省略旧图片输入。
8. `runtime.run_streaming(message, message_content=...)` 的 `message` 只含 OCR 与用户问题，且用户问题位于末尾；`message_content` 才含 Base64。因此短期记忆持久化 OCR、问题、最终回答和图片描述，但不持久化图片数据。
9. 构造 `report` 闭包流式处理 `thinking` / `agent_reasoning` / `agent_text` / `subagent_text`；正文开始后折叠 reasoning 面板。
10. `runtime.memory_queue.flush()` 后提交最终卡片；任何异常回统一错误提示。

#### 13.2.3 卡片渲染

- 单个 `message_id` 对应一个 `_running_cards[message_id] = card_id`。群聊第一次回复按事件 `chat_id` 调用 `im.v1.message.create`，保证卡片出现在原群主消息流；私聊使用 `reply_card`。中间文本进入每消息唯一的 coalescing worker，约 300ms 合并并串行调用 `_update_card`；最终答案必须等待 pending patch 全部完成后再提交，禁止旧片段覆盖最终卡片。
- `final=True` 时清掉本 message 的 running card 与上次文本缓存。
- 卡片 content 是 `interactive` 类型的 markdown 元素（`build_card_content`）。

#### 13.2.4 内容解析

`parse_feishu_content(content)`:
- 直接 `text` 字段 → 文本。
- `file_key` / `image_key` 顶层 → 占位 `[file]` / `[image]`，并把 key 收进 files。
- 富文本 `content: list[paragraph]`：每个 paragraph 是元素列表，元素 tag 为 `text`/`at` 取 `text`，`img` / `file` / `media` 收 key 并占位。

`should_trigger_agent` / `clean_mention_text` 显式处理 `@` 字符与 mentions 列表，避免误把消息体里的人名当作触发信号。

### 13.3 `FeishuThreadStore`

JSON 文件，路径来自 `Settings.feishu_thread_store_path`。

| 顶层字段（每个 entry） | 含义 |
| --- | --- |
| key (`feishu:<chat_id>:<topic_id>`) | 唯一索引。 |
| `thread_id` | `feishu_<16hex>`。 |
| `user_id`, `chat_id`, `topic_id` | 反查关联。 |
| `created_at` / `updated_at` | epoch float。 |

写入路径用 `tempfile.NamedTemporaryFile` + `Path.replace` 做原子替换；并发由 `threading.Lock` 保护。

### 13.4 `FeishuChannelService.run_forever`

注册 SIGINT/SIGTERM 触发 stop_event；`await stop_event.wait()` 阻塞主协程。`main()` 是 console script 入口，先 `load_dotenv()`、配 INFO logging。

### 13.5 `WeComChannel` / `AIEngineClient`

`channels/wecom.py` 使用官方 `wecom-aibot-python-sdk` 的 `WSClient` 接收 `text/voice/mixed` 消息和进入会话事件。它不直接构建 `AgentRuntime`，而由 `channels/ai_engine_client.py` 调用 `/internal/chat` 并解析 SSE，从而复用 AI Engine 内唯一的 Memory/LightRAG 生命周期。

- 单聊用户键：`wecom:<bot_id>:<sender_userid>`，thread scope 为 sender userid。
- 群聊用户键：`wecom-group:<bot_id>:<chat_id>`，thread scope 固定为 `__group__`，所有群成员共享上下文、Memory 和 RAG 状态。
- `wecom_user_id_map` 单聊查 sender userid，群聊查 `chat:<chatid>`；命中后改用映射的网页 user id，身份变化时 store 轮换 thread。
- `WeComThreadStore` 原子持久化 thread id 和每会话 RAG 开关。
- 同会话用 `asyncio.Lock` 串行，全局用 `Semaphore` 限流；回调 msgid 用有界 TTL 缓存去重。
- `/rag on|off|status` 不进入模型，直接读写通道状态。
- 流式回复先在 5 秒内发送准备状态，再按 `stream_interval_ms` 合并更新，`done.answer` 作为最终内容。
- 图片/文件提示用户走 Knowledge 上传；语音使用企业微信转写文本。

完整管理后台步骤与运维约束见 [`channels/WECOM.md`](superassist/channels/WECOM.md)。

### 13.6 `WeComRPAChannel`

`channels/wecom_rpa.py` 面向企业微信 5.x 中由普通微信用户创建的外部群。由于消息控件不暴露 UIA，它通过 `pywinauto` 截图、RapidOCR 和气泡区域分割读取当前可见群聊，再复用 `/internal/chat`。安全门按固定顺序执行：标题含“外部群” -> 群名白名单 -> 消息以前缀唤醒 -> 持久化去重 -> Agent；发送前再次 OCR 校验群名。窗口最小化、切到私聊/非白名单群、无法识别发送按钮时一律停止发送。

- 只监控当前打开的群，不自动点击会话列表。
- 群身份默认为 `wecom-rpa-group:<group-name-hash>`，thread scope 固定为 `__group__`。
- `wecom_user_id_map` 可用 `rpa:<群名>` 映射到网页知识空间。
- `.superassist/channels/wecom_rpa_state.json` 保存 24 小时可见消息指纹，启动时先 prime，避免回复历史消息。
- RPA 使用浅色主题的气泡颜色定位；客户端升级或主题变化后必须重新做只读识别验证。

---

<a id="14-uiserverpy"></a>
## 14. `ui/server.py`

该模块同时保留两个 FastAPI app factory：`create_app` 是向后兼容的独立记忆图查看器；`create_ai_engine_app` 是 Go 调用的内部 AI Engine。产品路径使用后者，浏览器不应直接访问内部路由。

### 14.1 路由

| 路径 | 行为 |
| --- | --- |
| `GET /api/graph?user_id=...&update_limit=80` | 返回 `{nodes, edges, updates, stats}`；update_limit clamp `[1,500]`。 |
| `GET /api/subagents/tasks?limit=50` | 返回 `{tasks: [SubagentResult.to_dict(), ...]}`，limit clamp `[1,200]`。 |
| `GET /api/subagents/tasks/{task_id}` | 单任务详情；缺失 → 404。 |
| `DELETE /api/subagents/tasks/{task_id}` | 从内存 store 删除；缺失 → 404。 |
| `GET /` | 返回 `frontend/index.html`。 |
| 静态挂载 `/` | `frontend/` 目录全量。 |

AI Engine 内部路由：

| 路径 | 行为 |
| --- | --- |
| `GET /internal/health` | Go 健康检查。 |
| `POST /internal/chat` | 接收 `user_id/message/thread_id/rag_mode`，返回 SSE 事件流。 |
| `GET /internal/graph` | 按 Go 注入的 `user_id` 返回长期记忆图。 |
| `GET/PUT /internal/settings` | 读取、校验并写回 Memory/飞书/企业微信配置；secret 只写不回显。 |
| `GET/POST/DELETE /internal/rag/documents...` | LightRAG 文档列表、上传和删除。 |
| `GET /internal/rag/graph` | 当前用户知识图。 |

`_sse_chat_stream` 在线程中运行同步 `AgentRuntime.run_streaming`，通过 `asyncio.Queue` 把回调事件桥接回 FastAPI 事件循环；`finally` 中 flush Memory queue、关闭 runtime，并发送 sentinel，保证每个请求只有一个终止事件。

### 14.2 `graph_payload`

- 节点列表通过 `_node_payload(node, recall_snapshot.get(node.id))` 序列化；命中 recall 的节点附带 `active_recall=True` + `recall_tier` + `recall_score` + `recall_components` + `recall_updated_at`。
- `_limit_recall_snapshot`：`recall_snapshot` 总条目超过 `memory_top_k` 时按 (`tier_order`, `-score`) 排序后只保留前 `memory_top_k` —— UI 显示与 prompt 注入保持一致条数。
- `updates`: `_node_updates` + `_edge_updates` 合并，按 `updated_at` 倒序取 `update_limit` 条。
- `stats`: `{nodes, edges, by_type: {event,concept,intent,time → count}}`。

### 14.3 启动

`run_server(host, port, user_id)` 对应兼容命令 `superassist-memory-ui`。`serve_ai_engine(host, port)` 对应产品命令 `superassist-ai-engine`，默认只监听 `127.0.0.1:8765`，由 Go 的 `SUPERASSIST_PYTHON_HOST` 指向它。

---

<a id="15-clipy"></a>
## 15. `cli.py`

| 参数 | 默认 | 行为 |
| --- | --- | --- |
| `message` (positional, nargs=*) | — | 单 turn 调用必填。 |
| `--user-id` | `local-user` | 透传给 runtime。 |
| `--thread-id` | `None` | 缺省时 runtime 生成 `thread_<12hex>`。 |
| `-i/--interactive` | `False` | 进入交互 REPL；不要求 `message`。 |
| `--flush-memory` | `False` | 退出前调 `runtime.memory_queue.flush()`。 |

`_run_interactive`：thread_id 缺省时立即生成（**整段会话共用同一个 thread**）；`exit` / `quit` 退出；EOF / Ctrl-C 清退；最终按需 flush。

`_print_tool_event`：把 `agent_tool_call` 的 content 当文本直接打印；`tool_start` 打 `[tool:start] <name> {args}`；`tool_result` 打 `[tool:<status>] <name> error=<...>`。`AgentRuntime` 默认带这个 reporter，所以 CLI 即用即看到工具流。

---

<a id="16-cross-module-sequence"></a>
## 16. 跨模块时序：一条用户消息从进入到落库

下面以 lead agent 一次同步 `run(message, user_id, thread_id)` 为例，把上述所有模块的协作顺序串清楚：

1. **CLI / 飞书 / Python SSE 入口** 调 `AgentRuntime.run*` → `_initial_state(message, user_id, thread_id)` 加载 `messages.jsonl` 与 `thread_meta.json`，组装初始 `SuperAssistState`；Web 路径还携带 `rag_mode`。
2. **`team_thread_context(thread_id)`** 写入 `ContextVar`，使 `team_task` 后续可拿到 thread。
3. **`agent.invoke(state, ...)`** 进入 LangChain 内核：
   - `before_agent`：`MemoryRecallMiddleware` 调 `MemoryService.prepare_turn_contexts(...)`：embed 用户消息 → 重建 FAISS → 算 read/write 桶 → 写 recall snapshot → touch 节点 → **预分配**本 turn EVENT ID（实际节点由 writer 创建）→ 把 read/write recall 写回 state。
   - RAG 模式第二个 `before_agent`：`RagRetrievalMiddleware` 用原问题执行 `mix`，把结构化证据和来源写入 state/session。
4. 模型循环（每次模型调用前后会触发 wrap_*/before_*/after_*）：
   - `wrap_model_call`（`DynamicContextMiddleware`）拼 system prompt 前缀（recall + skills + time + 可选 RAG 证据/规则）；`ToolEventMiddleware` 在 model 返回后扫 `tool_calls` 并报 `agent_tool_call` 事件。
   - 模型若提出工具调用：`wrap_tool_call` 链由外向内 `ToolErrorMiddleware → ToolCallLimitMiddleware → ToolEventMiddleware` 包裹真正的工具实现。
   - `after_model`：`SubagentLimitMiddleware` 修剪超过 `max_concurrent` 的 `task` 调用；RAG 模式下 `RagRetryMiddleware` 在无证据且模型准备结束时注入下一次 `rag_search`。
5. **工具实现**：
   - `task` → `SubagentExecutor` 起 graph → 内部 `create_agent`（无 middleware）流式跑 → reporter 透传 `subagent_text`。
   - `team_task` → `TeamSupervisor.invoke` → `TeamLedger` 写 task 记录（hash + sig 链验证）→ `TeamMember` 在自己的 `AsyncLoopThread` 上拿/建 `ACPSession` → 收到响应 → 写 outbox + result 链。
   - `read_file(path="/mnt/skills/...")` → `ToolEventMiddleware` 自动把 skill name 加入 `loaded_skills`。
   - `rag_search` → 当前 `RagTurnSession` → `LightRAGService.retrieve`，共享最多 N 次尝试额度。
   - 网络/文件/shell 工具走各自的 sandbox 与开关。
6. **`after_agent`**（反向）：
   - `RagAttributionMiddleware`（仅 RAG）：根据 session 和工具事件生成来源尾注与 provenance metadata。
   - `FinalTextMiddleware`：把最后一条 AIMessage 文本写到 `metadata.final_assistant_text` + `memory_ready=True`。
   - `MemoryWriterMiddleware`：构造 `MemoryWritePayload` 入 `MemoryWriteQueue`（debounce 30s 后或 CLI flush 时批写）。
   - `ShortMemoryMiddleware`：只把本 turn 的 user/final-assistant 追加到 `messages.jsonl`；活动段达到 30 回合或 80000 tokens 时，用独立模型整体生成新 summary 并推进 metadata 检查点，不改写 JSONL。
7. **`AgentRuntime._result_from`** 读 `metadata.final_assistant_text` 与 `loaded_skills` 输出 `AgentRunResult`。
8. **后台**：`MemoryWriteQueue` 倒计时到点 → `MemoryWriter.write(payload)`：`_build_plan` (LLM 或 fallback) → `apply_plan` → `consolidate`（merge concepts / decay edges / complete orphans）；写完触发 `rebuild_vector_index`，FAISS 与 SQLite 一致性恢复。
9. **UI / 飞书**：
   - 飞书每收到 `agent_text` / `subagent_text` 事件就 patch 卡片；最终 `final=True` 时清缓存。
   - React 通过 Go `/api/graph` 拉 `nodes/edges/updates/stats`；聊天完成事件触发会话和记忆图自动刷新。
   - Knowledge 页面轮询文档状态，并通过 `/api/rag/graph` 单独显示 LightRAG 实体关系图。

各模块不互相调用对方的内部细节——所有跨边界数据都经过本文档列出的 Pydantic / dataclass 契约。修改任一字段时，按"持有者 → 中间件 → 入口"的顺序回放本时序，是判断"会不会破坏哪一段"的最快方法。

---

<a id="17-rag"></a>
## 17. `rag/`

本节只记录与其它 Python 模块交界的合同；切片、抽取、同名实体聚合、五种检索模式、删除语义和上游 LightRAG 默认值见 [`superassist/rag/README.md`](superassist/rag/README.md)。

### 17.1 `documents.py`

`SUPPORTED_EXTENSIONS` 为 `.txt/.md/.json/.csv/.html/.htm/.pdf/.docx/.pptx/.xlsx`。`extract_document(path)` 统一返回纯文本：PDF 用 pypdf，Office 文件用各自解析库，HTML 丢弃 script/style/noscript，CSV/XLSX 用 ` | ` 保留表格列。`safe_filename` 去掉路径并替换 Windows 非法字符，原始客户端文件名不能直接作为磁盘路径。

### 17.2 `service.LightRAGService`

- 构造时创建专用 asyncio loop/thread，普通同步 Agent 通过 `run_coroutine_threadsafe` 调用 LightRAG，避免同一个异步锁跨 loop。
- `base_dir=settings.rag_dir`；用户目录是 `sha256(user_id)[:24]`，所有 public 方法都要求 user_id。
- `upload` 先写随机 `doc-<uuid>` 文件和 `documents.json` manifest，再后台执行 `extract_document → rag.ainsert`。
- 每用户只缓存一个 `LightRAG` 实例；embedding 复用 SuperAssist embedder，LLM 复用 `create_chat_model(settings)`。
- `retrieve` 只调用 `aquery_data` 并转换成 `RagRetrievalResult`，最终回答仍由 lead Agent 生成。
- `delete` 调 `adelete_by_doc_id` 后才删除原文件和 manifest；失败状态保留错误便于 UI 展示。
- `graph` 读取上游 graph storage 的全部节点/边，过滤悬空边并按 degree/weight 归一化给前端。
- `close` 停止 LightRAG worker、finalize storages、停止专用 loop；AI Engine lifespan 必须调用它。

### 17.3 `context.RagTurnSession`

每轮通过 `rag_turn_context` 写入 `ContextVar`。Session 保存 `user_id/enabled/max_attempts/attempts/queries/sources/successful` 并用锁保护 `search`。工具只从 ContextVar 取 session，因此没有模型可控的 user_id 参数。检索异常转换成 `RagRetrievalResult(success=False)`，不让知识库故障终止整个 Agent。

### 17.4 HTTP 与 Go 边界

[`ui/rag.py`](superassist/ui/rag.py) 暴露 `/internal/rag/*`，Go 的 JWT handler 才暴露 `/api/rag/*`。上传接口必须同时执行：浏览器/Go 请求体限制、Python 文件数量/单文件大小限制、扩展名检查和 `safe_filename`。任何新增内部 RAG 路由都必须在 Go proxy 中显式映射，不能依赖浏览器直连 Python。
