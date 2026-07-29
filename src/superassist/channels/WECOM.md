# 企业微信智能机器人接入

SuperAssist 使用企业微信官方智能机器人 WebSocket 长连接 SDK。企业微信主动把消息推送到本机适配器，适配器再调用现有 Python AI Engine 的 `/internal/chat` SSE 接口。因此不需要公网回调地址、固定公网 IP 或自行处理回调加解密，也不会创建第二套 Agent、Memory 或 LightRAG 实例。

## 运行结构

```text
企业微信客户端
  -> 企业微信 WebSocket 服务
  -> superassist-wecom
  -> http://127.0.0.1:8765/internal/chat (SSE)
  -> Agent + CogniFold Memory + Tools/Subagents + optional LightRAG
  -> 企业微信流式 Markdown 回复
```

机器人进程和 AI Engine 必须在同一台机器上运行，或将 `SUPERASSIST_WECOM_AI_ENGINE_URL` 指向受信任的 AI Engine 地址。不要让多个 Python 进程直接构造并写入同一份本地 LightRAG 存储。

## 一、企业微信后台操作

1. 注册并登录 [企业微信管理后台](https://work.weixin.qq.com/)，创建一个用于开发测试的企业组织。内部测试不要求先完成企业认证，但未认证组织的部分外部联系能力会受限。
2. 使用管理员账号进入管理后台，找到“智能机器人”并创建机器人。后台菜单名称可能随企业微信版本调整，可从“应用管理”或“安全与管理”中的智能机器人入口进入。
3. 在机器人的 API 模式中选择 **WebSocket 长连接**，不要选择需要公网 URL 的回调模式。
4. 保存后取得机器人 `Bot ID` 和 `Secret`。Secret 只在服务端配置，不要提交到 Git、截图或发送到群聊。
5. 设置机器人可用范围并保存。测试时先只开放给自己的企业微信账号，验证完成后再逐步扩大范围。
6. 在企业微信客户端中进入机器人单聊，或将机器人添加到允许的群聊。

官方文档：

- [智能机器人长连接开发指南](https://developer.work.weixin.qq.com/document/path/101463)
- [消息回调协议](https://developer.work.weixin.qq.com/document/path/100719)
- [流式消息回复](https://developer.work.weixin.qq.com/document/path/101031)
- [Python SDK](https://pypi.org/project/wecom-aibot-python-sdk/)

## 二、SuperAssist 配置

先安装当前项目，使 `superassist-wecom` 命令和官方 SDK 进入 `CF` 环境：

```powershell
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist
python -m pip install -e .
```

可在 Web 前端的 **Settings -> WeCom** 页面填写，也可直接编辑 `.env`：

```dotenv
SUPERASSIST_WECOM_BOT_ID=后台复制的BotID
SUPERASSIST_WECOM_BOT_SECRET=后台复制的Secret
SUPERASSIST_WECOM_ALLOWED_USER_IDS=zhangsan,lisi
SUPERASSIST_WECOM_USER_ID_MAP={"zhangsan":"user_个人空间ID","chat:群聊chatid":"user_群知识空间ID"}
SUPERASSIST_WECOM_RAG_MODE_DEFAULT=false
SUPERASSIST_WECOM_MAX_CONCURRENT=3
SUPERASSIST_WECOM_STREAM_INTERVAL_MS=300
SUPERASSIST_WECOM_AI_ENGINE_URL=http://127.0.0.1:8765
```

| 配置 | 作用 |
| --- | --- |
| `BOT_ID` / `BOT_SECRET` | 官方长连接认证凭据，必填。前端只显示“已配置”，不会回显 Secret。 |
| `ALLOWED_USER_IDS` | 逗号分隔的企业微信 `userid` 白名单；留空表示允许机器人可见范围内的所有成员。正式使用建议显式配置。 |
| `USER_ID_MAP` | JSON 对象。单聊键使用企业微信 `userid`；群聊键使用 `chat:<chatid>`。映射后可与指定网页用户共享 Memory 和 LightRAG；Settings 页面会显示当前网页登录用户 ID。 |
| `RAG_MODE_DEFAULT` | 新企业微信会话是否默认检索用户上传的 LightRAG 知识库。 |
| `MAX_CONCURRENT` | 所有企业微信会话同时执行的 Agent 请求上限。 |
| `STREAM_INTERVAL_MS` | 合并模型增量后向企业微信更新回复的最小间隔，防止过于频繁。 |
| `AI_ENGINE_URL` | 适配器调用的 Python AI Engine 地址，默认仅本机。 |

企业微信连接参数保存后，需要重启 `superassist-wecom`。Memory 参数对后续新请求生效。

## 三、启动与验证

至少启动两个终端：

```powershell
# 终端 1：Agent、Memory 和 RAG 服务
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist
superassist-ai-engine --port 8765
```

```powershell
# 终端 2：企业微信长连接适配器
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist
superassist-wecom
```

看到 `WeCom bot authenticated` 后，在企业微信中向机器人发送“你好”。首条回复先显示“正在准备上下文”，随后原地更新为流式回答。Go 服务不是企业微信聊天的必需链路，但使用网页上传知识库、查看图谱或修改 Settings 时仍应启动 Go 服务。

RAG 命令按聊天框持久化：单聊各自独立，群聊内所有成员共享同一个开关。

```text
/rag on       开启上传资料检索
/rag off      关闭上传资料检索
/rag status   查看当前状态
```

## 身份、会话与可靠性

- 单聊按发送者隔离，默认身份为 `wecom:<bot_id>:<sender_userid>`，不同联系人不共享上下文或长期记忆。
- 群聊按 `chat_id` 共享，默认身份为 `wecom-group:<bot_id>:<chat_id>`；所有群成员共用 thread、长期 Memory 和 RAG 开关。
- 单聊可用 `{"zhangsan":"web_user_id"}` 映射网页个人空间；群聊必须显式使用 `{"chat:group_chat_id":"web_user_id"}` 映射群知识空间，不会默认借用任意成员的个人资料。
- 修改会话身份映射会自动生成新 thread，避免把映射前的历史泄漏到映射后的网页账号。
- 映射和 RAG 开关保存在 `.superassist/channels/wecom_threads.json`，使用临时文件原子替换。
- 同一会话的请求串行执行，不同会话受全局并发上限约束。
- WebSocket 重连由官方 SDK 负责，适配器另有 10 分钟重复消息过滤，避免回调重试造成重复 Agent 执行。
- 图片和文件不会直接进入聊天 Agent；知识文件仍通过网页 Knowledge 页面上传，以保留文档状态、来源追溯和删除语义。语音消息使用企业微信提供的转写文本。

## 故障排查

| 现象 | 检查项 |
| --- | --- |
| 启动时报缺少 `BOT_ID` | Settings 保存后重启通道，确认运行目录为仓库根目录。 |
| 认证失败 | 重新复制 Bot ID/Secret，检查是否误用了企业应用的 Corp ID/Secret。 |
| 回复提示 AI Engine 不可用 | 先启动 `superassist-ai-engine --port 8765`，并核对 `AI_ENGINE_URL`。 |
| 某用户提示未授权 | 把该成员的企业微信 `userid` 加入 `ALLOWED_USER_IDS`，不是手机号、昵称或 open_id。 |
| RAG 开启但没有资料证据 | 单聊映射企业微信 userid；群聊映射 `chat:<chatid>` 到资料所在网页 user ID，并等待 Knowledge 文档状态变为 `ready`。 |
| 启动提示端口 8765 被占用 | 已有 AI Engine 在运行，不要重复启动；企业微信适配器本身不监听本地端口。 |

生产环境应把 AI Engine 继续限制在本机或受控内网，使用进程管理器自动拉起两个进程，限制 `.env` 和 `.superassist` 的文件权限，并定期备份 SQLite、线程与渠道映射数据。

## 普通微信外部群：桌面 RPA

官方智能机器人无法加入由普通微信用户创建的任意群。`superassist-wecom-rpa` 是独立的 Windows 桌面适配器：使用已登录的企业微信员工号留在外部群，通过本地截图、OCR 和键鼠自动化收发消息，再调用同一个 AI Engine。

在 Settings -> WeCom -> Desktop RPA for external groups 配置：

```dotenv
SUPERASSIST_WECOM_RPA_ALLOWED_GROUPS=项目答疑群,客户交流群
SUPERASSIST_WECOM_RPA_TRIGGER_PREFIXES=@SuperAssist,小助手
SUPERASSIST_WECOM_RPA_POLL_INTERVAL_SECONDS=1.5
SUPERASSIST_WECOM_RPA_REPLY_MAX_CHARS=3000
```

`ALLOWED_GROUPS` 使用界面顶部显示的完整群名，逗号分隔；不能为空。`TRIGGER_PREFIXES` 只匹配消息开头，例如“`小助手 Graph RAG 是什么`”；不能为空。群聊 RAG 默认值与 AI Engine 地址复用 `SUPERASSIST_WECOM_RAG_MODE_DEFAULT`、`SUPERASSIST_WECOM_AI_ENGINE_URL`。如需让群共享网页上传的知识空间，在现有 `SUPERASSIST_WECOM_USER_ID_MAP` 中加入 `{"rpa:项目答疑群":"网页用户ID"}`。

启动前把企业微信切到浅色主题，打开目标外部群并保持窗口未最小化，然后运行：

```powershell
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist
superassist-ai-engine --port 8765
```

另开终端：

```powershell
conda activate CF
Set-Location F:\CODE\SuperAssist\superAssist\SuperAssist
superassist-wecom-rpa
```

RPA 只读取当前打开的群，不会自动点击会话列表。以下任一情况都会暂停且不调用 Agent：当前窗口是私聊、标题下没有“外部群”、群名不在白名单、消息未以唤醒词开头、窗口最小化。模型返回后还会再次核对群名；运行期间切换会话时，待发送回答会被丢弃而不是发到新会话。

该入口依赖非官方桌面自动化能力，微信或企业微信升级、主题/DPI/布局变化可能导致识别失效，也存在平台风控风险。应使用专门的测试员工号、控制回复频率，并在升级客户端后先观察日志完成只读验证。
