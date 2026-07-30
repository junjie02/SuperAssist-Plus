# SuperAssist

SuperAssist 是一个面向长期协作场景的多智能体个人助理。系统使用 CogniFold 风格类型图维护跨会话长期记忆，使用 LightRAG 对用户上传的资料进行图检索与原文检索，并提供流式聊天、工具调用、Subagent、ACP Agent Teams、飞书/企业微信接入和运行时配置界面。

当前技术栈：**React + Vite、Go + Gin、Python + FastAPI、LangChain/LangGraph、SQLite/MySQL、FAISS、LightRAG、WebSocket/SSE**。

## 系统架构

```text
Browser (React)
    │ HTTP / WebSocket
    ▼
Go Gateway :8080
    ├── JWT、用户与会话 API
    ├── 前端静态文件
    └── 将聊天/图谱/设置/RAG 请求代理到 Python
              │ HTTP / SSE
              ▼
       Python AI Engine :8765
          ├── LangChain Agent + middleware
          ├── CogniFold Memory
          ├── LightRAG Knowledge Base
          ├── Tools / Subagents / ACP Teams
          ├── Feishu channel
          └── WeCom channel -> AI Engine SSE

持久化：
  .superassist/superassist.sqlite3   业务数据与长期记忆图
  .superassist/faiss/                长期记忆向量入口索引
  .superassist/rag/<user-hash>/      每用户 LightRAG 文件、图和向量数据
  .superassist/threads/              对话 JSONL 与短记忆摘要
```

Go 是面向浏览器的产品入口：负责认证、WebSocket 生命周期、会话列表、静态资源以及 Python 内部 API 的安全代理。Python 只监听本机地址，负责模型、Memory、RAG 和 Agent 执行。

## 主要功能

- **流式聊天**：WebSocket 连接浏览器与 Go，Go 消费 Python SSE，并实时转发思考、工具和回答事件。
- **长期记忆**：`event/concept/intent/time` 四类节点、九类边，结合向量入口、BFS、Personalized PageRank、时间和访问频次召回。
- **LightRAG 知识库**：批量上传 PDF、DOCX、PPTX、XLSX、TXT、Markdown、JSON、CSV 和 HTML，异步完成切片、实体关系抽取和建图。
- **Agentic RAG**：首轮默认 `mix` 检索；证据不足时允许模型改写查询，并由中间件保证最多完成三轮检索。失败后可按工具配置联网，最终回答明确标注上传资料、网页或模型知识来源。
- **工具与 Subagent**：支持文件、网页和可选 Shell 工具，通过 `task` 把复杂任务交给 general-purpose 或 research 子 Agent。
- **ACP Agent Teams**：通过 `agent_team.toml` 接入 Claude Code 等外部 Agent，使用带文件锁、hash chain 和 HMAC 的 JSONL ledger 记录协作过程。
- **可视化与管理**：导航栏提供 Chat、Memory Graph、Knowledge、Settings；管理员额外拥有 Users 页面，可查看各 Web/飞书/企业微信身份的会话、消息记录与长期记忆图，并可删除选中的未压缩短记忆记录。
- **飞书 Bot**：支持可配置的群聊触发、合并且保序的流式卡片更新及会话映射；图片与用户问题在同一次主 Agent 多模态请求中处理，可选本地 OCR 仅作校对辅助，主模型会在回答中附加可复用的针对性图片描述；后续历史只保留文本而不重复发送 Base64 图片。私聊按用户隔离，群聊按 `chat_id` 共享短记忆、长期 Memory 图和 RAG。
- **企业微信 Bot**：使用官方 WebSocket SDK，无需公网回调；支持流式回复、成员白名单、单聊隔离、群聊共享记忆、RAG 开关、重复消息过滤和断线重连。

## 快速开始

项目默认在 Windows、PowerShell 和 Conda 环境 `CF` 下开发。

```powershell
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist

# Python
python -m pip install -e .

# React
Set-Location frontend
npm install
npm run build
Set-Location ..
```

复制配置并填写模型地址和密钥：

```powershell
Copy-Item .env.example .env
```

分别启动两个后端进程：

```powershell
# 终端 1，项目根目录
superassist-ai-engine --port 8765

# 终端 2
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist\go-server
go run .
```

浏览器访问 `http://localhost:8080`。首次使用先注册用户。

> `go run .` 不能写成 `run .`。如果看到 `bind: Only one usage...` 或 `[Errno 10048]`，表示 8080/8765 已有进程监听，不要重复启动。Go 日志中的 Python `connection refused` 表示 AI Engine 尚未运行或端口不一致。

### 前端开发模式

先启动 Python 和 Go，再启动 Vite：

```powershell
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist\frontend
npm run dev
```

访问 `http://localhost:5173`。Vite 会把 `/api` 和 `/ws` 代理到 `127.0.0.1:8080`。

### CLI 与聊天渠道

```powershell
# 单轮
superassist "你好" --flush-memory

# 连续对话
superassist -i --thread-id my-thread --flush-memory

# 飞书通道
superassist-feishu

# 企业微信通道（需先启动 AI Engine）
superassist-wecom

# 企业微信桌面 RPA（普通微信外部群，需保持目标群窗口可见）
superassist-wecom-rpa
```

`superassist-memory-ui` 是仍保留的旧调试入口，不属于当前 React 产品的验证路径；完整界面应通过 Go 的 `http://localhost:8080` 使用。

## 知识库使用

1. 打开 **Knowledge** 页面并批量选择文件。
2. 上传接口立即返回 `202`；页面会轮询 `queued → parsing → indexing → ready/failed`。
3. 文档变为 `ready` 后，可在 Knowledge 页面查看知识图。
4. 在 Chat 输入框旁开启 RAG 模式后提问。
5. 删除文档会异步删除 LightRAG 中对应的原文、chunk、向量以及只由该文档支持的图数据。

当前 LightRAG 使用固定 Token 切片，默认约为 `1200 tokens + 100 overlap`；默认查询模式为 `mix`，将实体图、关系图和原文向量召回合并。完整实现说明见 [LightRAG 技术设计](src/superassist/rag/README.md)。

## 配置

Python 配置集中在 `src/superassist/config.py`，示例见 [.env.example](.env.example)。设置页写回项目根 `.env`；Memory 参数对新请求立即生效，飞书/企业微信连接参数需要重启对应通道。官方智能机器人与只服务普通微信外部群的桌面 RPA 配置、启动和排障见 [企业微信接入指南](src/superassist/channels/WECOM.md)。

### 核心配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SUPERASSIST_MODEL` | `gpt-4o-mini` | 主 Agent 与 LightRAG 使用的模型 |
| `SUPERASSIST_REASONING_EFFORT` | `medium` | GPT-5.6 推理强度；飞书可用 `/effort` 按会话覆盖 |
| `SUPERASSIST_MODEL_INPUT_LOG_ENABLED` | `false` | 将最终模型请求 payload 追加到 `logs/model-input.jsonl` |
| `SUPERASSIST_MODEL_INPUT_LOG_MAX_BYTES` | `52428800` | 模型输入日志单文件轮转阈值，保留 3 个备份 |
| `SUPERASSIST_FEISHU_IMAGE_OCR_ENABLED` | `true` | 飞书图片启用本地 RapidOCR 辅助；原图无论 OCR 是否成功都会直接交给主模型 |
| `SUPERASSIST_FEISHU_IMAGE_OCR_MAX_CHARS` | `12000` | 单条飞书多图 OCR 写入当前轮文本历史的字符上限 |
| `SUPERASSIST_FEISHU_IMAGE_CONTEXT_TTL_SECONDS` | `180` | 飞书会话原图上下文的滑动 TTL；有效消息会刷新，过期后只复用文本描述 |
| `SUPERASSIST_API_KEY` | 空 | 为空时使用测试用 fallback 模型 |
| `SUPERASSIST_BASE_URL` | OpenAI API | OpenAI 兼容接口地址 |
| `SUPERASSIST_DATA_DIR` | `.superassist` | SQLite、FAISS、RAG、线程等数据根目录 |
| `SUPERASSIST_DB_URL` | 空 | Python MySQL DSN；为空使用 SQLite |
| `SUPERASSIST_EMBEDDING_PROVIDER` | `bge` | `bge` 或测试用 `hash` |
| `SUPERASSIST_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | Memory 与 LightRAG 共用的向量模型 |
| `SUPERASSIST_ENABLE_TOOLS` | `false` | 常规工具总开关 |
| `SUPERASSIST_TOOL_NETWORK_ENABLED` | `true` | 是否允许 web search/fetch |
| `SUPERASSIST_TOOL_SHELL_ENABLED` | `false` | 是否允许 Shell |
| `SUPERASSIST_MEMORY_LLM_WRITER_ENABLED` | `true` | 使用独立 Memory Updater 写长期记忆图；关闭时走规则 writer |
| `SUPERASSIST_MEMORY_MODEL` | `deepseek-v4-flash` | Memory Updater 和短记忆压缩模型 |
| `SUPERASSIST_MEMORY_API_KEY` | 空 | 独立模型密钥；为空时复用主模型密钥 |
| `SUPERASSIST_MEMORY_BASE_URL` | 空 | 独立模型 OpenAI 兼容地址；为空时复用主模型地址 |
| `SUPERASSIST_MEMORY_TOP_K` | `12` | 注入模型的长期记忆节点总数 |
| `SUPERASSIST_RAG_MAX_ATTEMPTS` | `3` | 每轮聊天最多上传资料检索次数 |
| `SUPERASSIST_RAG_TOP_K` | `20` | LightRAG 实体/关系候选数 |
| `SUPERASSIST_RAG_CHUNK_TOP_K` | `10` | LightRAG 原文 chunk 候选数 |


短期记忆按完整回合追加，不使用滑动窗口。活动段达到 `30` 个已完成回合或约 `80000` tokens 时，由独立模型把“上一份摘要 + 整个活动段”压成新摘要，并从新的空活动段继续累计。`messages.jsonl` 始终追加，摘要通过 `thread_meta.json` 检查点切换；后续模型上下文只保留用户消息与最终助手回复，不保留工具参数或原始结果。

模型输入日志的 `call_kind` 可区分 `lead_agent`、`memory_updater` 和 `short_memory_compactor`；`input_manifest.component_tokens` 按消息角色与用途给出近似 token 分布，`input_manifest.sections` 进一步拆出短记忆摘要、长期记忆、运行时信息、RAG 和激活 Skill 等 XML 区段。

### Go 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SUPERASSIST_GO_PORT` | `8080` | 浏览器访问端口 |
| `SUPERASSIST_PYTHON_HOST` | `http://127.0.0.1:8765` | Python AI Engine 地址 |
| `SUPERASSIST_JWT_SECRET` | 开发默认值 | JWT 签名密钥，生产环境必须修改 |
| `SUPERASSIST_JWT_EXPIRY_HOURS` | `48` | 登录有效期 |
| `SUPERASSIST_ADMIN_USERNAMES` | `painting` | 逗号分隔的 Web 管理员用户名；Go 启动时同步管理员角色 |

MySQL URL 在 Python/SQLAlchemy 中使用 `mysql+pymysql://...`，Go/GORM 使用 Go MySQL DSN。当前默认的单机产品形态使用同一个 SQLite 文件；切换数据库前应同时确认 Go 与 Python 的连接格式。

## 数据与隔离

```text
.superassist/
├── superassist.sqlite3
├── faiss/
│   ├── <safe-user-id>.index
│   └── <safe-user-id>.mapping.json
├── rag/<sha256(user-id)[:24]>/
│   ├── documents.json
│   ├── files/
│   └── index/default/
├── threads/<thread-id>/
│   ├── messages.jsonl
│   └── thread_meta.json
├── teams/<thread-id>/ledger.jsonl
├── channels/feishu_threads.json
├── channels/wecom_threads.json
├── huggingface/
└── workspace/
```

长期记忆图和 LightRAG 知识图是两套独立数据：前者记录对话事件、概念、意图和时间关系；后者只记录用户上传资料的实体、关系与原文来源。

## Skill 渐进加载

Skill 采用三层渐进式加载，避免把整个技能库长期塞进模型上下文：

1. 注册器只扫描 `skills/public/*/SKILL.md` 一层，并始终注入名称、简介和路径索引。
2. 用户意图匹配时，Agent 读取主 `SKILL.md`；同一轮直接使用工具结果，后续轮次在激活期内注入主文件。
3. `references/`、`examples/`、脚本和资源不会递归注册，仅在任务需要时按路径读取；读取任意技能资源会刷新激活时间。

激活期由 `SUPERASSIST_SKILL_ACTIVE_TTL_SECONDS` 控制，默认 `300` 秒，也可在 Settings → Skills 修改。超时后线程只保留全局索引，不再注入完整 Skill 内容。

## 项目结构

```text
SuperAssist/
├── frontend/                 React + Vite，Chat/Graph/Knowledge/Users/Settings
├── go-server/                Gin 网关、认证、线程、WS 和 Python 代理
├── src/superassist/
│   ├── agent/                Agent 工厂、状态、运行时和短记忆
│   ├── memory/               CogniFold 图、向量入口和 PPR/BFS 排名
│   ├── rag/                  LightRAG 文档、索引、检索和技术文档
│   ├── middlewares/          Memory、RAG、工具和最终文本中间件
│   ├── subagents/            进程内子 Agent
│   ├── acp_client/           ACP 客户端
│   ├── teams/                ACP 团队与可审计 ledger
│   ├── channels/             飞书与企业微信通道
│   ├── tools/                文件、网页、Shell、task、team_task
│   └── ui/                   Python 内部 FastAPI API
├── skills/                   SKILL.md 技能
├── tests/                    Python 自动化测试
├── agent_team.toml           ACP 团队配置
└── pyproject.toml
```

## API 边界

浏览器只访问 Go：

- `/api/auth/*`：注册、登录和当前用户。
- `/api/threads/*`：会话列表、历史和删除。
- `/api/admin/users/*`：管理员查看所有身份、会话、消息历史和长期记忆图；普通用户访问返回 `403`。
- `/api/graph`：长期记忆图。
- `/api/rag/documents`、`/api/rag/graph`：知识库和 LightRAG 图。
- `/api/settings`：Memory、Skills、飞书与企业微信设置。
- `/ws/chat?token=...`：流式聊天。

Python `/internal/*` 仅供 Go 在本机调用，不应直接暴露到公网。

## 测试与构建

```powershell
# Python
python -B -m pytest
python -m ruff check src tests

# Go
Set-Location go-server
go test ./...
Set-Location ..

# Frontend
Set-Location frontend
npm run build
```

RAG 单测和详细运维命令见 [src/superassist/rag/README.md](src/superassist/rag/README.md)。

## 从 SuperAssist-Plus 迁移

当前 SQLite schema 与 FAISS 文件布局兼容旧项目。迁移前停止所有 SuperAssist 进程，并备份目标 `.superassist`：

```powershell
$sourceDir = "F:\CODE\SuperAssist\superAssist\SuperAssist-Plus\.superassist-plus"
$targetDir = "F:\CODE\SuperAssist\superAssist\SuperAssist\.superassist"

Copy-Item "$sourceDir\superassist_plus.sqlite3" "$targetDir\superassist.sqlite3"
Copy-Item -Recurse "$sourceDir\faiss" "$targetDir\faiss"
Copy-Item -Recurse "$sourceDir\threads" "$targetDir\threads"
Copy-Item -Recurse "$sourceDir\teams" "$targetDir\teams"
Copy-Item -Recurse "$sourceDir\channels" "$targetDir\channels"
```

LightRAG 是当前项目新增的数据域，需要通过 Knowledge 页面重新上传资料建立索引。

## License

MIT
