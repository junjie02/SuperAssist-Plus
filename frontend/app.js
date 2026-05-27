const graphSvg = document.querySelector("#graph");
const emptyState = document.querySelector("#emptyState");
const updateList = document.querySelector("#updateList");
const refreshButton = document.querySelector("#refreshButton");
const typeFilter = document.querySelector("#typeFilter");
const userIdInput = document.querySelector("#userId");

const detailsTitle = document.querySelector("#detailsTitle");
const detailsDescription = document.querySelector("#detailsDescription");

const SVG_NS = "http://www.w3.org/2000/svg";
const MIN_ZOOM = 0.35;
const MAX_ZOOM = 3;
const DRAG_THRESHOLD = 3;

const typeColors = {
  event: "#64748b",
  concept: "#2563eb",
  intent: "#059669",
  time: "#b45309",
};

let graphData = { nodes: [], edges: [], updates: [], stats: {} };
let selectedId = "";
let viewport = { x: 0, y: 0, scale: 1 };
let nodePositions = new Map();
let viewportLayer = null;
let edgeLayer = null;
let nodeLayer = null;
let renderedNodeMap = new Map();
let dragState = null;
let panState = null;

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function shortText(value, max = 42) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function displayScore(node) {
  return typeof node.recall_score === "number" ? node.recall_score : node.importance;
}

function scoreLabel(node) {
  return node.active_recall ? "Score" : "importance";
}

function setStats(stats) {
  document.querySelector("#nodeCount").textContent = stats.nodes ?? 0;
  document.querySelector("#edgeCount").textContent = stats.edges ?? 0;
  document.querySelector("#conceptCount").textContent = stats.by_type?.concept ?? 0;
  document.querySelector("#intentCount").textContent = stats.by_type?.intent ?? 0;
  document.querySelector("#updatedAt").textContent = `Loaded ${new Date().toLocaleTimeString()}`;
}

function visibleGraph() {
  const filter = typeFilter.value;
  const nodes =
    filter === "all" ? graphData.nodes : graphData.nodes.filter((node) => node.type === filter);
  const ids = new Set(nodes.map((node) => node.id));
  return {
    nodes,
    edges: graphData.edges.filter((edge) => ids.has(edge.source_id) && ids.has(edge.target_id)),
  };
}

function createSvgElement(name) {
  return document.createElementNS(SVG_NS, name);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function pointerToSvgPoint(event) {
  const point = graphSvg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  const matrix = graphSvg.getScreenCTM();
  if (!matrix) return { x: event.offsetX, y: event.offsetY };
  return point.matrixTransform(matrix.inverse());
}

function pointerToGraphPoint(event) {
  const point = pointerToSvgPoint(event);
  return {
    x: (point.x - viewport.x) / viewport.scale,
    y: (point.y - viewport.y) / viewport.scale,
  };
}

function applyViewportTransform() {
  if (!viewportLayer) return;
  viewportLayer.setAttribute(
    "transform",
    `translate(${viewport.x}, ${viewport.y}) scale(${viewport.scale})`,
  );
}

function pruneNodePositions(nodes) {
  const currentIds = new Set(nodes.map((node) => node.id));
  for (const id of nodePositions.keys()) {
    if (!currentIds.has(id)) nodePositions.delete(id);
  }
}

function layoutNodes(nodes, width, height) {
  if (!nodes.length) return [];
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.max(120, Math.min(width, height) * 0.34);
  const grouped = ["event", "concept", "intent", "time"].flatMap((type) =>
    nodes.filter((node) => node.type === type),
  );
  return grouped.map((node, index) => {
    const angle = (Math.PI * 2 * index) / grouped.length - Math.PI / 2;
    const ring = radius * (0.72 + (index % 3) * 0.14);
    const saved = nodePositions.get(node.id);
    if (saved) {
      return {
        ...node,
        x: saved.x,
        y: saved.y,
      };
    }
    const x = centerX + Math.cos(angle) * ring;
    const y = centerY + Math.sin(angle) * ring;
    nodePositions.set(node.id, { x, y });
    return {
      ...node,
      x,
      y,
    };
  });
}

function updateGraphPositions() {
  if (!nodeLayer || !edgeLayer) return;

  nodeLayer.querySelectorAll(".node").forEach((group) => {
    const node = renderedNodeMap.get(group.dataset.id);
    if (!node) return;
    group.setAttribute("transform", `translate(${node.x}, ${node.y})`);
  });

  edgeLayer.querySelectorAll(".edge").forEach((group) => {
    const source = renderedNodeMap.get(group.dataset.sourceId);
    const target = renderedNodeMap.get(group.dataset.targetId);
    if (!source || !target) return;
    const line = group.querySelector(".edge-line");
    const label = group.querySelector(".edge-label");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    label.setAttribute("x", String((source.x + target.x) / 2));
    label.setAttribute("y", String((source.y + target.y) / 2 - 6));
  });
}

function renderGraph() {
  const { nodes, edges } = visibleGraph();
  const width = Math.max(graphSvg.clientWidth, 720);
  const height = Math.max(graphSvg.clientHeight, 520);
  const laidOut = layoutNodes(nodes, width, height);
  const nodeMap = new Map(laidOut.map((node) => [node.id, node]));
  renderedNodeMap = nodeMap;

  emptyState.hidden = nodes.length > 0;
  graphSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  graphSvg.innerHTML = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#93a4b8"></path>
      </marker>
    </defs>
  `;

  viewportLayer = createSvgElement("g");
  viewportLayer.classList.add("viewport-layer");
  edgeLayer = createSvgElement("g");
  edgeLayer.classList.add("edge-layer");
  nodeLayer = createSvgElement("g");
  nodeLayer.classList.add("node-layer");
  viewportLayer.append(edgeLayer, nodeLayer);
  graphSvg.append(viewportLayer);
  applyViewportTransform();

  for (const edge of edges) {
    const source = nodeMap.get(edge.source_id);
    const target = nodeMap.get(edge.target_id);
    if (!source || !target) continue;
    const group = createSvgElement("g");
    group.classList.add("edge");
    group.dataset.id = edge.id;
    group.dataset.sourceId = edge.source_id;
    group.dataset.targetId = edge.target_id;

    const line = createSvgElement("line");
    line.classList.add("edge-line");
    if (selectedId === edge.id) line.classList.add("selected");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("stroke-width", String(1.2 + edge.weight * 2.4));

    const label = createSvgElement("text");
    label.classList.add("edge-label");
    label.setAttribute("x", String((source.x + target.x) / 2));
    label.setAttribute("y", String((source.y + target.y) / 2 - 6));
    label.setAttribute("text-anchor", "middle");
    label.textContent = edge.edge_type;

    group.append(line, label);
    group.addEventListener("click", () => selectEdge(edge, source, target));
    edgeLayer.append(group);
  }

  for (const node of laidOut) {
    const group = createSvgElement("g");
    group.classList.add("node");
    if (selectedId === node.id) group.classList.add("selected");
    if (node.active_recall) group.classList.add("active-recall");
    if (node.recall_tier) group.classList.add(`tier-${node.recall_tier}`);
    group.dataset.id = node.id;
    group.setAttribute("transform", `translate(${node.x}, ${node.y})`);

    const circle = createSvgElement("circle");
    circle.classList.add(node.type);
    circle.setAttribute("r", String(18 + Math.min(12, displayScore(node) * 12)));
    circle.setAttribute("fill", typeColors[node.type] || "#64748b");

    const label = createSvgElement("text");
    label.setAttribute("y", "38");
    label.setAttribute("text-anchor", "middle");
    label.textContent = shortText(node.title, 28);

    group.append(circle, label);
    group.addEventListener("pointerdown", (event) => startNodeDrag(event, node));
    nodeLayer.append(group);
  }
}

function startNodeDrag(event, node) {
  event.preventDefault();
  event.stopPropagation();
  const graphPoint = pointerToGraphPoint(event);
  dragState = {
    id: node.id,
    startX: graphPoint.x,
    startY: graphPoint.y,
    nodeStartX: node.x,
    nodeStartY: node.y,
    pointerStartX: event.clientX,
    pointerStartY: event.clientY,
    moved: false,
    element: event.currentTarget,
  };
  event.currentTarget.classList.add("dragging");
  graphSvg.setPointerCapture(event.pointerId);
}

function startCanvasPan(event) {
  if (event.button !== 0) return;
  if (event.target.closest(".node, .edge")) return;
  const point = pointerToSvgPoint(event);
  panState = {
    startX: point.x,
    startY: point.y,
    viewportX: viewport.x,
    viewportY: viewport.y,
    pointerStartX: event.clientX,
    pointerStartY: event.clientY,
    moved: false,
  };
  graphSvg.classList.add("is-panning");
  graphSvg.setPointerCapture(event.pointerId);
}

function handlePointerMove(event) {
  if (dragState) {
    const node = renderedNodeMap.get(dragState.id);
    if (!node) return;
    const graphPoint = pointerToGraphPoint(event);
    const deltaX = graphPoint.x - dragState.startX;
    const deltaY = graphPoint.y - dragState.startY;
    const movedPixels = Math.hypot(
      event.clientX - dragState.pointerStartX,
      event.clientY - dragState.pointerStartY,
    );
    dragState.moved = dragState.moved || movedPixels > DRAG_THRESHOLD;
    node.x = dragState.nodeStartX + deltaX;
    node.y = dragState.nodeStartY + deltaY;
    nodePositions.set(node.id, { x: node.x, y: node.y });
    updateGraphPositions();
    return;
  }

  if (panState) {
    const point = pointerToSvgPoint(event);
    const movedPixels = Math.hypot(
      event.clientX - panState.pointerStartX,
      event.clientY - panState.pointerStartY,
    );
    panState.moved = panState.moved || movedPixels > DRAG_THRESHOLD;
    viewport.x = panState.viewportX + point.x - panState.startX;
    viewport.y = panState.viewportY + point.y - panState.startY;
    applyViewportTransform();
  }
}

function finishPointerInteraction(event) {
  let nodeToSelect = null;
  if (dragState) {
    if (!dragState.moved) {
      nodeToSelect = renderedNodeMap.get(dragState.id);
    }
    dragState.element?.classList.remove("dragging");
    dragState = null;
  }
  if (panState) {
    panState = null;
    graphSvg.classList.remove("is-panning");
  }
  if (graphSvg.hasPointerCapture(event.pointerId)) {
    graphSvg.releasePointerCapture(event.pointerId);
  }
  if (nodeToSelect) {
    selectNode(nodeToSelect);
  }
}

function handleGraphWheel(event) {
  if (!renderedNodeMap.size) return;
  event.preventDefault();
  const point = pointerToSvgPoint(event);
  const graphPoint = {
    x: (point.x - viewport.x) / viewport.scale,
    y: (point.y - viewport.y) / viewport.scale,
  };
  const zoomFactor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  const nextScale = clamp(viewport.scale * zoomFactor, MIN_ZOOM, MAX_ZOOM);
  viewport.x = point.x - graphPoint.x * nextScale;
  viewport.y = point.y - graphPoint.y * nextScale;
  viewport.scale = nextScale;
  applyViewportTransform();
}

function selectNode(node) {
  selectedId = node.id;
  detailsTitle.textContent = node.title;
  const components = node.recall_components
    ? ` · PR ${node.recall_components.pagerank.toFixed(2)} · recency ${node.recall_components.recency.toFixed(2)} · access ${node.recall_components.access.toFixed(2)} · urgency ${node.recall_components.urgency.toFixed(2)}`
    : "";
  const tier = node.recall_tier ? ` · ${node.recall_tier}` : "";
  detailsDescription.textContent = `${node.type}${tier} · ${scoreLabel(node)} ${displayScore(node).toFixed(2)}${components} · ${node.description}`;
  renderGraph();
}

function selectEdge(edge, source, target) {
  selectedId = edge.id;
  detailsTitle.textContent = `${edge.edge_type} · ${edge.weight.toFixed(2)}`;
  detailsDescription.textContent = `${source.title} → ${target.title}. Updated ${formatTime(edge.updated_at)}.`;
  renderGraph();
}

function renderUpdates() {
  updateList.innerHTML = "";
  if (!graphData.updates.length) {
    updateList.innerHTML = '<article class="update-item"><h3>No updates yet</h3><p>The graph has no node or edge updates for this user.</p></article>';
    return;
  }
  for (const item of graphData.updates) {
    const article = document.createElement("article");
    article.className = "update-item";
    article.innerHTML = `
      <div class="update-row">
        <span class="badge ${item.kind}">${item.kind}</span>
        <p>${formatTime(item.updated_at)}</p>
      </div>
      <h3>${item.title}</h3>
      <p>${item.description}</p>
    `;
    updateList.append(article);
  }
}

async function loadGraph() {
  refreshButton.disabled = true;
  try {
    const userId = encodeURIComponent(userIdInput.value.trim() || "local-user");
    const response = await fetch(`/api/graph?user_id=${userId}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    graphData = await response.json();
    selectedId = "";
    pruneNodePositions(graphData.nodes || []);
    setStats(graphData.stats || {});
    renderGraph();
    renderUpdates();
  } catch (error) {
    updateList.innerHTML = `<article class="update-item"><h3>Load failed</h3><p>${error}</p></article>`;
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", loadGraph);
typeFilter.addEventListener("change", renderGraph);
userIdInput.addEventListener("change", loadGraph);
window.addEventListener("resize", renderGraph);
graphSvg.addEventListener("pointerdown", startCanvasPan);
graphSvg.addEventListener("pointermove", handlePointerMove);
graphSvg.addEventListener("pointerup", finishPointerInteraction);
graphSvg.addEventListener("pointercancel", finishPointerInteraction);
graphSvg.addEventListener("wheel", handleGraphWheel, { passive: false });

loadGraph();
