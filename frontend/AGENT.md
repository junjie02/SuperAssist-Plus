# Frontend Technical Documentation

IMPORTANT: Any change to the memory graph viewer UI, its data contract, visual
layout, interaction behavior, or local serving assumptions must update this
document.

## Purpose

The `frontend` directory contains the static memory graph viewer for
SuperAssist-Plus. It visualizes the SQLite-backed long-term memory graph as
nodes and typed edges, with a right-side scrolling activity panel showing recent
node and edge updates.

## Files

- `index.html`: application shell and semantic regions.
- `styles.css`: visual system, responsive layout, graph styling, and update
  panel styling.
- `app.js`: API loading, graph layout, SVG rendering, selection behavior,
  filtering, graph pan/zoom/node dragging, and update feed rendering.

## Runtime

The frontend is served by the FastAPI app in `superassist_plus.ui.server`
through the `superassist-plus-memory-ui` command. It expects the local API endpoint
`/api/graph?user_id=<id>` to return:

- `nodes`: memory nodes with id, type, title, description, importance, metadata,
  and timestamps.
- Active read-recall nodes also include `active_recall`, `recall_tier`,
  `recall_score`, `recall_components`, and `recall_updated_at`. The graph uses
  `recall_score` as the displayed node score for highlighted nodes.
- `edges`: typed memory edges with source, target, weight, metadata, and
  timestamps.
- `updates`: recent node/edge updates sorted newest-first.
- `stats`: aggregate counts for the current user graph.

## Design Notes

- Keep the UI dense and operational rather than marketing-oriented.
- Use restrained color, clear hierarchy, and stable graph dimensions.
- The graph canvas supports pointer-drag panning, wheel zooming, and direct node
  dragging. Node positions are kept in browser state across graph rerenders and
  filters until the page is reloaded.
- Nodes selected for the latest main-agent read recall are highlighted. Their
  circle size and detail panel score use the latest dynamic `Score(v)`, while
  non-highlighted nodes continue to show static `importance`.
- Avoid frontend build tooling unless the UI becomes complex enough to justify
  it.
- Do not introduce network-loaded assets; the viewer should work offline once
  the Python server is running.
