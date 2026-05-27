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
const LAYOUT_MARGIN = 96;
const TYPE_ORDER = ["event", "concept", "intent", "time"];

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
let pinnedNodes = new Set();

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
  for (const id of pinnedNodes.keys()) {
    if (!currentIds.has(id)) pinnedNodes.delete(id);
  }
}

function hashNumber(value) {
  let hash = 2166136261;
  const text = String(value || "");
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function typeAnchor(node, centerX, centerY, spreadX, spreadY) {
  switch (node.type) {
    case "event":
      return { x: centerX - spreadX * 0.34, y: centerY + spreadY * 0.08 };
    case "concept":
      return { x: centerX - spreadX * 0.04, y: centerY };
    case "intent":
      return { x: centerX + spreadX * 0.32, y: centerY + spreadY * 0.04 };
    case "time":
      return { x: centerX + spreadX * 0.18, y: centerY - spreadY * 0.32 };
    default:
      return { x: centerX, y: centerY };
  }
}

function initialNodePosition(node, index, centerX, centerY, spreadX, spreadY) {
  const saved = nodePositions.get(node.id);
  if (saved) return { x: saved.x, y: saved.y };

  const anchor = typeAnchor(node, centerX, centerY, spreadX, spreadY);
  const hash = hashNumber(`${node.type}:${node.id}`);
  const angle = ((hash % 360) * Math.PI) / 180;
  const ring = 42 + (index % 9) * 18 + (hash % 31);
  return {
    x: anchor.x + Math.cos(angle) * ring,
    y: anchor.y + Math.sin(angle) * ring,
  };
}

function edgeGeometry(source, target, curveOffset = 0) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const normalX = -dy / distance;
  const normalY = dx / distance;
  const midX = (source.x + target.x) / 2;
  const midY = (source.y + target.y) / 2;
  const controlX = midX + normalX * curveOffset;
  const controlY = midY + normalY * curveOffset;

  return {
    path: `M ${source.x} ${source.y} Q ${controlX} ${controlY} ${target.x} ${target.y}`,
    labelX: midX + normalX * curveOffset * 0.72,
    labelY: midY + normalY * curveOffset * 0.72 - 8,
  };
}

function edgePairKey(edge) {
  return [edge.source_id, edge.target_id].sort().join("::");
}

function edgeCurveOffsets(edges) {
  const totals = new Map();
  const seen = new Map();
  const offsets = new Map();

  for (const edge of edges) {
    const key = edgePairKey(edge);
    totals.set(key, (totals.get(key) || 0) + 1);
  }

  for (const edge of edges) {
    const key = edgePairKey(edge);
    const total = totals.get(key) || 1;
    const index = seen.get(key) || 0;
    seen.set(key, index + 1);
    const baseOffset = total > 1 ? (index - (total - 1) / 2) * 28 : ((hashNumber(edge.id) % 5) - 2) * 5;
    offsets.set(edge.id, baseOffset);
  }

  return offsets;
}

function layoutNodes(nodes, edges, width, height) {
  if (!nodes.length) return [];
  const centerX = width / 2;
  const centerY = height / 2;
  const graphScale = Math.sqrt(Math.max(nodes.length, 1));
  const spreadX = Math.max(width * 0.84, graphScale * 250, 760);
  const spreadY = Math.max(height * 0.82, graphScale * 210, 560);
  const minX = centerX - spreadX / 2 + LAYOUT_MARGIN;
  const maxX = centerX + spreadX / 2 - LAYOUT_MARGIN;
  const minY = centerY - spreadY / 2 + LAYOUT_MARGIN;
  const maxY = centerY + spreadY / 2 - LAYOUT_MARGIN;
  const grouped = TYPE_ORDER.flatMap((type) =>
    nodes.filter((node) => node.type === type),
  );
  const laidOut = grouped.map((node, index) => {
    const initial = initialNodePosition(node, index, centerX, centerY, spreadX, spreadY);
    return {
      ...node,
      x: clamp(initial.x, minX, maxX),
      y: clamp(initial.y, minY, maxY),
      vx: 0,
      vy: 0,
    };
  });
  const nodeMap = new Map(laidOut.map((node) => [node.id, node]));
  const graphEdges = edges
    .map((edge) => ({
      ...edge,
      source: nodeMap.get(edge.source_id),
      target: nodeMap.get(edge.target_id),
    }))
    .filter((edge) => edge.source && edge.target);

  const iterations = Math.min(280, 80 + nodes.length * 7 + graphEdges.length * 2);
  const repulsion = 32000 + Math.min(nodes.length, 120) * 420;
  const springLength = clamp(185 + nodes.length * 1.4, 175, 260);
  const springStrength = 0.018;
  const anchorStrength = 0.008;
  const centerStrength = 0.003;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const forces = new Map(laidOut.map((node) => [node.id, { x: 0, y: 0 }]));

    for (let a = 0; a < laidOut.length; a += 1) {
      for (let b = a + 1; b < laidOut.length; b += 1) {
        const source = laidOut[a];
        const target = laidOut[b];
        let dx = target.x - source.x;
        let dy = target.y - source.y;
        let distanceSq = dx * dx + dy * dy;
        if (distanceSq < 1) {
          const angle = ((hashNumber(`${source.id}:${target.id}`) % 360) * Math.PI) / 180;
          dx = Math.cos(angle);
          dy = Math.sin(angle);
          distanceSq = 1;
        }
        const distance = Math.sqrt(distanceSq);
        const force = repulsion / Math.max(distanceSq, 900);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        forces.get(source.id).x -= fx;
        forces.get(source.id).y -= fy;
        forces.get(target.id).x += fx;
        forces.get(target.id).y += fy;
      }
    }

    for (const edge of graphEdges) {
      const source = edge.source;
      const target = edge.target;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const weight = clamp(Number(edge.weight) || 0.5, 0.2, 1.4);
      const force = (distance - springLength) * springStrength * weight;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      forces.get(source.id).x += fx;
      forces.get(source.id).y += fy;
      forces.get(target.id).x -= fx;
      forces.get(target.id).y -= fy;
    }

    for (const node of laidOut) {
      const force = forces.get(node.id);
      const anchor = typeAnchor(node, centerX, centerY, spreadX, spreadY);
      force.x += (anchor.x - node.x) * anchorStrength;
      force.y += (anchor.y - node.y) * anchorStrength;
      force.x += (centerX - node.x) * centerStrength;
      force.y += (centerY - node.y) * centerStrength;

      if (pinnedNodes.has(node.id)) continue;
      node.vx = (node.vx + force.x) * 0.72;
      node.vy = (node.vy + force.y) * 0.72;
      node.x = clamp(node.x + node.vx, minX, maxX);
      node.y = clamp(node.y + node.vy, minY, maxY);
    }
  }

  for (const node of laidOut) {
    nodePositions.set(node.id, { x: node.x, y: node.y });
    delete node.vx;
    delete node.vy;
  }

  return laidOut;
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
    const path = group.querySelector(".edge-line");
    const label = group.querySelector(".edge-label");
    const geometry = edgeGeometry(source, target, Number(group.dataset.curveOffset) || 0);
    path.setAttribute("d", geometry.path);
    label.setAttribute("x", String(geometry.labelX));
    label.setAttribute("y", String(geometry.labelY));
  });
}

function renderGraph() {
  const { nodes, edges } = visibleGraph();
  const width = Math.max(graphSvg.clientWidth, 720);
  const height = Math.max(graphSvg.clientHeight, 520);
  const laidOut = layoutNodes(nodes, edges, width, height);
  const nodeMap = new Map(laidOut.map((node) => [node.id, node]));
  const curveOffsets = edgeCurveOffsets(edges);
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
    if (selectedId === edge.id) group.classList.add("selected");
    if (selectedId === edge.source_id || selectedId === edge.target_id) group.classList.add("connected");
    group.dataset.id = edge.id;
    group.dataset.sourceId = edge.source_id;
    group.dataset.targetId = edge.target_id;
    group.dataset.curveOffset = String(curveOffsets.get(edge.id) || 0);

    const geometry = edgeGeometry(source, target, curveOffsets.get(edge.id) || 0);
    const line = createSvgElement("path");
    line.classList.add("edge-line");
    if (selectedId === edge.id) line.classList.add("selected");
    line.setAttribute("d", geometry.path);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke-width", String(1.1 + edge.weight * 2.3));

    const label = createSvgElement("text");
    label.classList.add("edge-label");
    label.setAttribute("x", String(geometry.labelX));
    label.setAttribute("y", String(geometry.labelY));
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
    if (dragState.moved) pinnedNodes.add(node.id);
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
