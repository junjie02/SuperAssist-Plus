# SuperAssist

SuperAssist 是一个面向长期协作场景的多智能体个人助理。系统使用 CogniFold 风格类型图维护跨会话长期记忆，使用 LightRAG 对用户上传的资料进行图检索与原文检索，并提供流式聊天、工具调用、Subagent、ACP Agent Teams、飞书接入和运行时配置界面。

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
          └── Feishu channel

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
- **可视化与设置**：导航栏提供 Chat、Memory Graph、Knowledge、Settings；设置页可管理 Memory 与飞书参数。
- **飞书 Bot**：支持私聊和群聊 @ 触发、流式卡片更新及会话映射。

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

### CLI 与飞书

```powershell
# 单轮
superassist "你好" --flush-memory

# 连续对话
superassist -i --thread-id my-thread --flush-memory

# 飞书通道
superassist-feishu
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

Python 配置集中在 `src/superassist/config.py`，示例见 [.env.example](.env.example)。设置页写回项目根 `.env`；Memory 参数对新请求立即生效，飞书连接参数需要重启飞书通道。

### 核心配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SUPERASSIST_MODEL` | `gpt-4o-mini` | 聊天、Memory writer 与 LightRAG 使用的模型 |
| `SUPERASSIST_API_KEY` | 空 | 为空时使用测试用 fallback 模型 |
| `SUPERASSIST_BASE_URL` | OpenAI API | OpenAI 兼容接口地址 |
| `SUPERASSIST_DATA_DIR` | `.superassist` | SQLite、FAISS、RAG、线程等数据根目录 |
| `SUPERASSIST_DB_URL` | 空 | Python MySQL DSN；为空使用 SQLite |
| `SUPERASSIST_EMBEDDING_PROVIDER` | `bge` | `bge` 或测试用 `hash` |
| `SUPERASSIST_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | Memory 与 LightRAG 共用的向量模型 |
| `SUPERASSIST_ENABLE_TOOLS` | `false` | 常规工具总开关 |
| `SUPERASSIST_TOOL_NETWORK_ENABLED` | `true` | 是否允许 web search/fetch |
| `SUPERASSIST_TOOL_SHELL_ENABLED` | `false` | 是否允许 Shell |
| `SUPERASSIST_MEMORY_LLM_WRITER_ENABLED` | `false` | LLM 写长期记忆图；关闭时走规则 writer |
| `SUPERASSIST_MEMORY_TOP_K` | `12` | 注入模型的长期记忆节点总数 |
| `SUPERASSIST_RAG_MAX_ATTEMPTS` | `3` | 每轮聊天最多上传资料检索次数 |
| `SUPERASSIST_RAG_TOP_K` | `20` | LightRAG 实体/关系候选数 |
| `SUPERASSIST_RAG_CHUNK_TOP_K` | `10` | LightRAG 原文 chunk 候选数 |

### Go 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SUPERASSIST_GO_PORT` | `8080` | 浏览器访问端口 |
| `SUPERASSIST_PYTHON_HOST` | `http://127.0.0.1:8765` | Python AI Engine 地址 |
| `SUPERASSIST_JWT_SECRET` | 开发默认值 | JWT 签名密钥，生产环境必须修改 |
| `SUPERASSIST_JWT_EXPIRY_HOURS` | `48` | 登录有效期 |

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
├── huggingface/
└── workspace/
```

长期记忆图和 LightRAG 知识图是两套独立数据：前者记录对话事件、概念、意图和时间关系；后者只记录用户上传资料的实体、关系与原文来源。

## 项目结构

```text
SuperAssist/
├── frontend/                 React + Vite，Chat/Graph/Knowledge/Settings
├── go-server/                Gin 网关、认证、线程、WS 和 Python 代理
├── src/superassist/
│   ├── agent/                Agent 工厂、状态、运行时和短记忆
│   ├── memory/               CogniFold 图、向量入口和 PPR/BFS 排名
│   ├── rag/                  LightRAG 文档、索引、检索和技术文档
│   ├── middlewares/          Memory、RAG、工具和最终文本中间件
│   ├── subagents/            进程内子 Agent
│   ├── acp_client/           ACP 客户端
│   ├── teams/                ACP 团队与可审计 ledger
│   ├── channels/             飞书通道
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
- `/api/graph`：长期记忆图。
- `/api/rag/documents`、`/api/rag/graph`：知识库和 LightRAG 图。
- `/api/settings`：Memory 与飞书设置。
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
