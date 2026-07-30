# Frontend Technical Documentation

Update this document whenever frontend routes, API contracts, refresh events, chat streaming, graph behavior, responsive layout, or serving assumptions change.

## Purpose And Runtime

`frontend/` is the authenticated React 18 + Vite application for SuperAssist. It provides Chat, Memory Graph, Knowledge, Users, and Settings views.

- Development: Vite serves `http://localhost:5173` and proxies `/api` plus `/ws` to Go at `127.0.0.1:8080`.
- Production: `npm run build` writes `frontend/dist`; Go serves those assets and handles SPA fallback.
- Trust boundary: browser code only calls Go. It never reads `.env`, the SQLite database, local RAG files, or Python `/internal/*` routes.
- Icons come from the bundled `lucide-react` package. Do not add CDN dependencies.

Commands:

```powershell
npm install
npm run dev
npm run build
```

## Navigation And Lifetime

[src/layouts/MainLayout.jsx](src/layouts/MainLayout.jsx) owns persistent navigation:

| Path | View | Lifetime |
| --- | --- | --- |
| `/` | Chat | Always mounted after login; hidden with CSS on other pages |
| `/graph` | Memory Graph | Always mounted; reloads on turn-completed events |
| `/knowledge` | Uploaded documents and LightRAG graph | Mounted while selected |
| `/users` | Admin-only user directory, conversations, transcript, and per-user memory graph | Mounted while selected |
| `/settings` | Memory, Feishu, and WeCom configuration | Mounted while selected |
| `/login` | Login/register | Public route |

Keeping Chat mounted is intentional: WebSocket, selected thread, draft state, and conversation history must survive navigation.

## Main Files

- `src/pages/ChatPage.jsx`: thread list, history, streaming chat, RAG toggle, and autosizing composer.
- `src/pages/GraphPage.jsx`: CogniFold memory graph/list modes and refresh behavior.
- `src/pages/KnowledgePage.jsx`: multipart upload, document state polling, deletion, and LightRAG graph.
- `src/pages/UsersPage.jsx`: admin user directory, per-user thread list, Markdown transcript, and memory graph viewer.
- `src/pages/SettingsPage.jsx`: validated Memory, Feishu, and WeCom settings editor.
- `src/components/GraphCanvas.jsx`: shared canvas renderer for both graph domains.
- `src/hooks/useAuth.jsx`: token and authenticated-user lifecycle.
- `src/hooks/useWebSocket.js`: reconnectable chat transport.
- `src/lib/api.js`: authenticated JSON/multipart API client.
- `src/lib/events.js`: cross-view event names.
- `src/lib/graph-layout.js`: stable force-layout helpers.
- `src/App.css`: design tokens, page layout, controls, and responsive rules.

## Public Go API Contracts

All protected HTTP calls use `Authorization: Bearer <token>`. The WebSocket sends the token as a query parameter because browser WebSocket construction cannot set an Authorization header.

### Chat And Threads

- `GET /api/threads`: conversation list derived from persisted thread files.
- `GET /api/threads/:id/history`: stored conversation records.
- `DELETE /api/threads/:id`: deletes the selected thread.
- `GET /ws/chat?token=...`: WebSocket chat transport.

Outgoing chat messages include `message`, optional `thread_id`, and `rag_mode: boolean`. Incoming events may include `preparing_context`, `thinking`, `agent_text`, tool/subagent events, `done`, or `error`.

The composer grows with explicit and wrapped lines through five visual lines, then keeps a stable maximum height and enables vertical scrolling. Reset its measured height after a message is sent or the input is cleared.

### Memory Graph

`GET /api/graph` returns:

```text
nodes[]  id/type/title/description/importance/access_count/metadata/recall fields
edges[]  source_id/target_id/edge_type/weight/metadata
updates[]
stats    nodes/edges/by_type
```

Active recall fields are `active_recall`, `recall_tier`, `recall_score`, `recall_components`, and `recall_updated_at`.

### Knowledge

- `GET /api/rag/documents`: per-user documents, supported extensions, and upload limits.
- `POST /api/rag/documents`: multipart `files`; returns 202 with queued documents.
- `DELETE /api/rag/documents/:id`: returns 202 and starts asynchronous deletion.
- `GET /api/rag/graph`: current user's LightRAG nodes, edges, stats, and `updated_at`.

Document states are `queued`, `parsing`, `indexing`, `ready`, `failed`, and `deleting`. Poll while any document is in a transient state. Do not infer LightRAG internal progress percentages from these states.

### Settings

`GET /api/settings` returns `memory`, `feishu`, `wecom`, and `meta`. Feishu and WeCom secrets are never returned; only `app_secret_configured` and `bot_secret_configured` are exposed.

`PUT /api/settings` accepts all three groups. Omitting a channel secret preserves it, while sending `""` clears it. The Python engine validates related numeric constraints and atomically updates the root `.env`. Memory values apply to newly created runtimes; Feishu/WeCom connection changes set channel-specific restart-required flags.

### Admin Users

- `GET /api/admin/users`: registered Web users plus virtual Feishu/WeCom identities and activity totals.
- `GET /api/admin/users/:user_id/threads`: conversations owned by one identity.
- `GET /api/admin/users/:user_id/threads/:thread_id/history`: transcript records with stable-for-that-response `record_index` values.
- `DELETE /api/admin/users/:user_id/threads/:thread_id/messages/:record_index`: remove one current JSONL short-memory record; this does not alter compressed summaries or long-term graph nodes.
- `GET /api/admin/users/:user_id/graph`: the selected identity's CogniFold memory graph.

The Users navigation and route are rendered only when `/api/auth/me` returns `is_admin: true`. This is convenience, not authorization: Go rechecks the persisted admin role for every `/api/admin/*` request.

## Refresh And Error Semantics

- A completed chat turn reloads the conversation list and dispatches `superassist:turn-completed`.
- Memory Graph listens for that event and reloads graph/list data after the Python chat path flushes memory writes.
- Knowledge polls documents while indexing/deleting and reloads the LightRAG graph after state changes.
- `api.js` checks content type before JSON parsing. An HTML SPA/error response must surface as an HTTP/API error, not `Unexpected token '<'`.
- A chat transport error must leave the current thread and composer usable for retry.

## Graph And Responsive Invariants

- Graph positions remain stable across rerenders and filters; dragged nodes are pinned for the browser session.
- Canvas dimensions must be stable so labels, loading states, and controls do not resize the graph container.
- The admin Users view offers an enlarged memory-graph dialog with the same pan, zoom, selection, and refresh behavior as the inline graph.
- Memory and knowledge graphs use the same renderer but different data semantics; never merge their node sets in frontend state.
- Desktop navigation is a left sidebar. Narrow screens use compact icon navigation and one-column settings fields.
- Labels and action controls must not overlap at supported mobile and desktop widths.
