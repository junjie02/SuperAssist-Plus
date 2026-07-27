# Frontend Technical Documentation

IMPORTANT: Any change to the frontend data contracts, visual layout,
interaction behavior, or local serving assumptions must update this document.

## Purpose

The `frontend` directory contains the React application for SuperAssist. It
provides authenticated chat, memory graph inspection, per-user LightRAG
document management, and runtime settings for memory and Feishu integration.

## Runtime

- Vite serves the application during development and proxies `/api` and `/ws`
  to the Go server on `127.0.0.1:8080`.
- Production assets are built into `frontend/dist` and served by the Go server.
- The Go server owns authentication and proxies AI operations to the Python
  engine. Browser code never accesses `.env` or the Python internal API.
- Icons come from the locally bundled `lucide-react` dependency; no frontend
  assets are loaded from a CDN.

## Main Files

- `src/layouts/MainLayout.jsx`: persistent navigation and page switching.
- `src/pages/ChatPage.jsx`: streamed chat and thread history.
- `src/pages/GraphPage.jsx`: graph/list memory views.
- `src/pages/SettingsPage.jsx`: memory and Feishu settings editor.
- `src/pages/KnowledgePage.jsx`: batch upload and LightRAG indexing status.
- `src/components/GraphCanvas.jsx`: interactive graph rendering.
- `src/lib/api.js`: authenticated JSON API client.
- `src/App.css`: shared visual system and responsive layouts.

## API Contracts

`GET /api/graph` returns `nodes`, `edges`, `updates`, and aggregate `stats`.
Active recall nodes also expose `active_recall`, `recall_tier`, `recall_score`,
`recall_components`, and `recall_updated_at`.

`GET /api/settings` returns:

- `memory`: all editable long-memory and short-memory values.
- `feishu`: App ID, domain, allowed Open IDs, mention-only mode, and
  `app_secret_configured`. The App Secret itself is never returned.
- `meta`: runtime application status.

`PUT /api/settings` accepts the `memory` and `feishu` groups. Omitting
`feishu.app_secret` preserves the current secret; an empty string clears it.
The Python engine validates related numeric values, writes the corresponding
`SUPERASSIST_*` keys to the project `.env`, and applies memory changes to new
requests. Feishu connection changes require the Feishu channel to restart.

`GET /api/rag/documents` returns the authenticated user's documents, supported
extensions, and upload limits. `POST /api/rag/documents` accepts multipart
`files` and returns queued documents; `DELETE /api/rag/documents/:id` starts an
asynchronous index deletion. The page polls while any item is queued, parsing,
indexing, or deleting.

WebSocket chat messages include `rag_mode: boolean`. In RAG mode the Python
agent performs LightRAG retrieval, may rewrite and retry queries up to the
configured limit, and appends explicit uploaded/web/model provenance.

## Design Notes

- Keep the interface dense and operational, with stable control dimensions.
- Chat remains mounted while users inspect graph, knowledge, or settings so its in-memory
  state is preserved during navigation.
- The chat composer grows with wrapped or explicit new lines up to five text
  lines, then keeps a stable height and enables vertical scrolling.
- A completed chat turn refreshes the conversation list and dispatches the
  `superassist:turn-completed` browser event. The mounted graph page listens for
  that event and reloads the shared Graph/List payload after memory is flushed.
- On narrow screens the sidebar becomes a compact icon navigation bar and the
  settings form changes from two columns to one.
- Graph node positions remain in browser state across rerenders and filters;
  manually dragged nodes are pinned for the session.
