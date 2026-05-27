# Channels Module Technical Documentation

IMPORTANT: Update this document when IM channel behavior, configuration,
thread mapping, or channel startup changes.

## Purpose

`channels/` owns lightweight IM integrations for SuperAssist-Plus. The first
channel is Feishu/Lark, implemented as a local WebSocket long-connection
service that calls `AgentRuntime` directly. It intentionally does not copy
DeerFlow's Gateway/LangGraph SDK channel stack.

## Feishu Behavior

- `superassist-plus-feishu` starts the Feishu WebSocket channel.
- Credentials come from `SUPERASSIST_PLUS_FEISHU_APP_ID` and
  `SUPERASSIST_PLUS_FEISHU_APP_SECRET`.
- Startup creates and caches one `AgentRuntime`, which preloads the configured
  embedder such as BGE before the first Feishu message arrives. Later messages
  reuse that runtime and only swap the per-message progress reporter.
- Feishu users map to SuperAssist users as `feishu:<open_id>`.
- Feishu `chat_id + topic_id` maps to one SuperAssist thread persisted under
  `{SUPERASSIST_PLUS_DATA_DIR}/channels/feishu_threads.json`.
- Private chats always trigger. Group chats trigger only on bot mentions when
  `SUPERASSIST_PLUS_FEISHU_MENTION_ONLY=true`.
- The first version supports text/rich-text messages only. File-only and
  image-only messages receive a short unsupported-message reply.
- The channel creates one running interactive card and patches it for
  context-preparation status, model-authored text progress, and the final
  answer. Tool call details are not shown in Feishu.
- Empty tool-call assistant text is ignored; the channel never converts raw tool
  names or arguments into progress text.

## Maintenance Notes

- Keep SDK-specific code behind `FeishuChannel`; keep parsing and routing
  helpers deterministic and independently testable.
- Do not add token streaming here unless `AgentRuntime` exposes a real stream.
