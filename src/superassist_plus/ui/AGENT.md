# UI Module Technical Documentation

IMPORTANT: Any change to local UI serving, graph API shape, frontend path
resolution, or memory visualization commands must update this document.

## Purpose

The `ui` module serves the static memory graph viewer and exposes a FastAPI
JSON API over the existing SQLite memory store.

## Files

- `server.py`: FastAPI app factory, `/api/graph` handler, static asset mounting,
  uvicorn runner, and CLI entrypoint.
- `__init__.py`: package marker.

## API

`GET /api/graph?user_id=local-user`

Returns:

- `nodes`: all memory nodes for the requested user.
- `edges`: all typed memory edges for the requested user.
- `updates`: recent node and edge updates sorted by `updated_at`.
- `stats`: total counts and node counts by type.

The API is read-only and uses `MemoryService`/`MemoryGraphStore` rather than
querying SQLite ad hoc from the frontend.

## Command

```powershell
superassist-plus-memory-ui --user-id local-user --port 8765
```

The command serves `frontend/` from the project root and prints the local URL.

## Maintenance Notes

- Keep the FastAPI surface small and typed; add routers only when the API grows
  beyond simple local inspection.
- Do not expose write endpoints without adding explicit tests and docs.
- Keep the frontend data contract synchronized with `frontend/AGENT.md`.
