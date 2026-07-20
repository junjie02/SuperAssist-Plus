# SuperAssist

AI 助手 Web 应用，支持流式对话、CogniFold 类型图谱长期记忆、工具调用、子智能体。

技术栈：**Go + Python + React + MySQL/SQLite + WebSocket + SSE**

## 架构

```
浏览器 ──HTTP/WS──→ Go (Gin) ──SSE──→ Python (FastAPI) → AI Agent
    (React)          :8080              :8765              (LangChain)
                         │
                         ▼
                      MySQL / SQLite
```

- **React 前端** — 登录/注册、流式聊天、记忆图谱查看
- **Go 服务** — JWT 认证、WebSocket 管理、线程 CRUD、静态文件、代理 AI 请求
- **Python AI 引擎** — LangChain/LangGraph Agent、记忆算法、FAISS 向量索引、工具执行
- **数据库** — 开发默认 SQLite，生产切 MySQL（`SUPERASSIST_DB_URL`）

## 快速开始

```powershell
conda activate CF
cd f:\CODE\SuperAssist\superAssist\SuperAssist

# 安装 Python 依赖
pip install -e .

# 构建前端
cd frontend && npm install && npm run build && cd ..

# 1. 启动 Python AI 引擎（后台）
superassist-ai-engine --port 8765

# 2. 启动 Go Web 服务（前台）
cd go-server && go run .
```

浏览器打开 `http://localhost:8080`，注册账号后即可使用。

### 开发模式（前端热更新）

```powershell
# 终端 1: Python AI 引擎
superassist-ai-engine --port 8765

# 终端 2: Go 服务
cd go-server && go run .

# 终端 3: React 前端（Vite dev server，自动代理 API 到 Go）
cd frontend && npm run dev
```

浏览器打开 `http://localhost:5173`。

### CLI 模式

```powershell
# 单轮对话
superassist "你好" --flush-memory

# 交互式多轮，--thread-id 保持跨次启动连续性
superassist -i --thread-id my-thread --flush-memory
```

### 记忆图谱可视化（独立模式）

```powershell
superassist-memory-ui --user-id local-user --port 8765
```

浏览器打开 `http://localhost:8765`。

### 飞书 Bot

```powershell
superassist-feishu
```

需要 `.env` 中配置 `SUPERASSIST_FEISHU_APP_ID` 和 `SUPERASSIST_FEISHU_APP_SECRET`。

## 功能

- **流式 AI 对话** — WebSocket 实时推送，支持思考过程展示、工具调用可视化
- **长期记忆** — CogniFold 类型图谱（event/concept/intent/time 四种节点 + 9 种边类型）
- **工具调用** — 文件读写、网页搜索、Shell 执行（可选开启）
- **子智能体** — `task` 工具可派发复杂任务给 general-purpose / research 子 Agent
- **ACP 团队** — 通过 `agent_team.toml` 配置 Claude Code 等外部 Agent 协作
- **记忆图谱查看器** — 节点/边可视化、力导向布局、拖拽缩放
- **飞书集成** — WebSocket Bot，支持群里 @ 触发

## 环境变量

所有配置使用 `SUPERASSIST_` 前缀，在项目根目录 `.env` 文件中设置。

### 模型

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_MODEL_PROVIDER` | `openai` | 模型提供商 |
| `SUPERASSIST_MODEL` | `gpt-4o-mini` | 模型名称 |
| `SUPERASSIST_API_KEY` | — | API 密钥，为空时使用本地 fallback（不联网） |
| `SUPERASSIST_BASE_URL` | `https://api.openai.com/v1` | API 地址 |
| `SUPERASSIST_TEMPERATURE` | — | 温度参数 |
| `SUPERASSIST_MAX_TOKENS` | — | 最大 token 数 |

### 数据库

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_DB_URL` | — | MySQL DSN，为空时使用 SQLite |
| `SUPERASSIST_DATA_DIR` | `.superassist` | 数据目录（SQLite、FAISS、线程文件） |

MySQL 连接格式：`mysql+pymysql://用户名:密码@主机:3306/数据库名`

### 服务端口

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_GO_PORT` | `8080` | Go Web 服务端口 |
| `SUPERASSIST_PYTHON_HOST` | `http://127.0.0.1:8765` | Python AI 引擎地址（Go 调用） |

### JWT 认证

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_JWT_SECRET` | — | JWT 签名密钥，生产环境必填 |
| `SUPERASSIST_JWT_EXPIRY_HOURS` | `48` | Token 过期时间（小时） |

### 工具

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_ENABLE_TOOLS` | `false` | 总开关 |
| `SUPERASSIST_TOOL_NETWORK_ENABLED` | `true` | 网络搜索/抓取 |
| `SUPERASSIST_TOOL_SHELL_ENABLED` | `false` | Shell 执行 |
| `SUPERASSIST_MAX_TOOL_CALLS` | `8` | 每轮最大工具调用次数 |

### 子智能体

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_SUBAGENTS_ENABLED` | `true` | 启用 task 工具 |
| `SUPERASSIST_SUBAGENT_MAX_CONCURRENT` | `3` | 最大并发子任务 |
| `SUPERASSIST_SUBAGENT_TIMEOUT_SECONDS` | `900` | 子任务超时 |
| `SUPERASSIST_SUBAGENT_MAX_TURNS` | `20` | 子任务最大轮次 |

### 记忆

完整列表参见 `.env` 文件的 `# --- Memory ---` 区块，包含读路径、写路径、巩固衰减、短记忆四组参数。

关键参数：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SUPERASSIST_MEMORY_TOP_K` | `12` | 每次注入 LLM 的记忆节点数 |
| `SUPERASSIST_MEMORY_LLM_WRITER_ENABLED` | `false` | LLM 写图谱（`true`=高质量，`false`=规则写） |
| `SUPERASSIST_EMBEDDING_PROVIDER` | `bge` | `bge`=中文语义向量，`hash`=确定性离线向量（测试用） |
| `SUPERASSIST_EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | Embedding 模型 |
| `SUPERASSIST_EMBEDDING_DEVICE` | `cpu` | Embedding 设备 |

### 记忆算法档位

**读路径**：控制怎么找回相关记忆

- `SUPERASSIST_MEMORY_READ_USE_PPR=true` — BFS + Personalized PageRank 混合（推荐）
- `SUPERASSIST_MEMORY_READ_USE_PPR=false` — 纯 BFS 扩散

**写路径**：控制怎么把新知识写入图谱

- `SUPERASSIST_MEMORY_LLM_WRITER_ENABLED=true` — LLM 分析对话，产出结构化更新（质量高，费 token）
- `SUPERASSIST_MEMORY_LLM_WRITER_ENABLED=false` — 规则写（省 token，离线可用）

更多参数及说明见 `.env` 文件内注释。

## 项目结构

```
SuperAssist/
├── frontend/              React + Vite SPA
│   ├── src/
│   │   ├── pages/         LoginPage, ChatPage, GraphPage
│   │   ├── layouts/       MainLayout（侧边栏 + 内容区）
│   │   ├── components/    可复用组件
│   │   ├── hooks/         useAuth, useWebSocket
│   │   └── lib/           api 封装
│   └── dist/              构建产物（Go 直接 serve）
├── go-server/             Go Web 服务（Gin + GORM）
│   ├── main.go            入口
│   ├── config/            环境变量解析
│   ├── handler/           auth, thread, graph 路由处理
│   ├── ws/                WebSocket 聊天
│   ├── middleware/         JWT 中间件
│   ├── proxy/             Python AI 引擎 HTTP 客户端
│   ├── service/           业务逻辑
│   └── model/             GORM 模型
├── src/superassist/       Python AI 引擎
│   ├── agent/             运行时、中间件链、prompt、流式
│   ├── memory/            类型图谱（存储、嵌入、评分、向量索引）
│   ├── middlewares/       9 条中间件（一条文件一个）
│   ├── acp_client/        ACP 协议客户端
│   ├── teams/             agent_team.toml 团队管理
│   ├── subagents/         子智能体执行器
│   ├── channels/          飞书 WebSocket 通道
│   ├── tools/             LangChain 工具函数
│   ├── skills/            SKILL.md 注册表
│   ├── ui/                FastAPI AI 引擎 + 记忆图谱后端
│   └── auth/              认证模块（预留，Go 侧实现）
├── skills/                技能定义（SKILL.md）
├── .env                   环境变量配置
└── pyproject.toml         Python 项目配置
```

## 测试

```powershell
python -B -m pytest                              # 全量
python -B -m pytest tests/test_memory.py         # 记忆模块
python -B -m pytest tests/test_memory.py -k merge -x  # 单个测试
```

## 从 SuperAssist-Plus 迁移数据

SQLite schema 和 FAISS 布局未变，可直拷：

```powershell
$src = "f:\CODE\SuperAssist\superAssist\SuperAssist-Plus\.superassist-plus"
$dst = "f:\CODE\SuperAssist\superAssist\SuperAssist\.superassist"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item -Recurse "$src\threads"  "$dst\threads"
Copy-Item       "$src\superassist_plus.sqlite3"  "$dst\superassist.sqlite3"
Copy-Item -Recurse "$src\faiss"    "$dst\faiss"
Copy-Item -Recurse "$src\teams"    "$dst\teams"
Copy-Item -Recurse "$src\channels" "$dst\channels"
```

## MySQL 部署

```powershell
# Docker（推荐）
docker run -d --name mysql-superassist `
  -e MYSQL_ROOT_PASSWORD=yourpassword `
  -e MYSQL_DATABASE=superassist `
  -p 3306:3306 `
  mysql:8.0

# .env 配置
SUPERASSIST_DB_URL=mysql+pymysql://root:yourpassword@localhost:3306/superassist
```

首次启动时自动建表。Go 管 `users` 表，Python 管 `memory_nodes`、`memory_edges` 等记忆表。

## 生产部署

```powershell
# 构建
cd frontend && npm run build && cd ..
cd go-server && go build -o superassist-server.exe && cd ..

# 启动
superassist-ai-engine --port 8765 &      # Python 后台
.\go-server\superassist-server.exe        # Go 前台，含前端静态文件
```

浏览器打开 `http://localhost:8080`。

## License

MIT
