// Force-directed graph layout — pure functions ported from app.js

const LAYOUT_MARGIN = 96
const TYPE_ORDER = ['event', 'concept', 'intent', 'time']

export const TYPE_COLORS = {
  event: '#64748b',
  concept: '#2563eb',
  intent: '#059669',
  time: '#b45309',
}

function hashNumber(value) {
  let hash = 2166136261
  const text = String(value || '')
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)) }

function typeAnchor(node, cx, cy, spreadX, spreadY) {
  switch (node.type) {
    case 'event':   return { x: cx - spreadX * 0.34, y: cy + spreadY * 0.08 }
    case 'concept': return { x: cx - spreadX * 0.04, y: cy }
    case 'intent':  return { x: cx + spreadX * 0.32, y: cy + spreadY * 0.04 }
    case 'time':    return { x: cx + spreadX * 0.18, y: cy - spreadY * 0.32 }
    default:        return { x: cx, y: cy }
  }
}

function initialPosition(node, idx, cx, cy, spreadX, spreadY, saved) {
  if (saved) return { x: saved.x, y: saved.y }
  const anchor = typeAnchor(node, cx, cy, spreadX, spreadY)
  const h = hashNumber(`${node.type}:${node.id}`)
  const angle = ((h % 360) * Math.PI) / 180
  const ring = 42 + (idx % 9) * 18 + (h % 31)
  return { x: anchor.x + Math.cos(angle) * ring, y: anchor.y + Math.sin(angle) * ring }
}

export function displayScore(node) {
  return typeof node.recall_score === 'number' ? node.recall_score : node.importance
}

export function shortText(value, max = 28) {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  return text.length > max ? text.slice(0, max - 1) + '…' : text
}

// --- Edge geometry -------------------------------------------------------

export function edgeGeometry(source, target, curveOffset = 0) {
  const dx = target.x - source.x
  const dy = target.y - source.y
  const dist = Math.max(1, Math.hypot(dx, dy))
  const nx = -dy / dist
  const ny = dx / dist
  const mx = (source.x + target.x) / 2
  const my = (source.y + target.y) / 2
  const cx = mx + nx * curveOffset
  const cy = my + ny * curveOffset
  return {
    path: `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`,
    labelX: mx + nx * curveOffset * 0.72,
    labelY: my + ny * curveOffset * 0.72 - 8,
  }
}

function edgePairKey(e) { return [e.source_id, e.target_id].sort().join('::') }

export function edgeCurveOffsets(edges) {
  const totals = new Map()
  const seen = new Map()
  const offsets = new Map()
  for (const e of edges) totals.set(edgePairKey(e), (totals.get(edgePairKey(e)) || 0) + 1)
  for (const e of edges) {
    const key = edgePairKey(e)
    const total = totals.get(key) || 1
    const idx = seen.get(key) || 0
    seen.set(key, idx + 1)
    offsets.set(e.id, total > 1 ? (idx - (total - 1) / 2) * 28 : ((hashNumber(e.id) % 5) - 2) * 5)
  }
  return offsets
}

// --- Force-directed layout ------------------------------------------------

export function layoutNodes(nodes, edges, width, height, opts = {}) {
  if (!nodes.length) return []

  const { pinned = new Set(), savedPositions = new Map() } = opts
  const cx = width / 2
  const cy = height / 2
  const graphScale = Math.sqrt(Math.max(nodes.length, 1))
  const spreadX = Math.max(width * 0.84, graphScale * 250, 760)
  const spreadY = Math.max(height * 0.82, graphScale * 210, 560)
  const minX = cx - spreadX / 2 + LAYOUT_MARGIN
  const maxX = cx + spreadX / 2 - LAYOUT_MARGIN
  const minY = cy - spreadY / 2 + LAYOUT_MARGIN
  const maxY = cy + spreadY / 2 - LAYOUT_MARGIN

  const grouped = TYPE_ORDER.flatMap(t => nodes.filter(n => n.type === t))
  const laidOut = grouped.map((node, i) => {
    const pos = initialPosition(node, i, cx, cy, spreadX, spreadY, savedPositions.get(node.id))
    return { ...node, x: clamp(pos.x, minX, maxX), y: clamp(pos.y, minY, maxY), vx: 0, vy: 0 }
  })

  const nodeMap = new Map(laidOut.map(n => [n.id, n]))
  const graphEdges = edges
    .map(e => ({ ...e, source: nodeMap.get(e.source_id), target: nodeMap.get(e.target_id) }))
    .filter(e => e.source && e.target)

  const iterations = Math.min(280, 80 + nodes.length * 7 + graphEdges.length * 2)
  const repulsion = 32000 + Math.min(nodes.length, 120) * 420
  const springLen = clamp(185 + nodes.length * 1.4, 175, 260)
  const springK = 0.018
  const anchorK = 0.008
  const centerK = 0.003

  for (let iter = 0; iter < iterations; iter++) {
    const forces = new Map(laidOut.map(n => [n.id, { x: 0, y: 0 }]))

    for (let a = 0; a < laidOut.length; a++) {
      for (let b = a + 1; b < laidOut.length; b++) {
        const sa = laidOut[a], sb = laidOut[b]
        let dx = sb.x - sa.x, dy = sb.y - sa.y
        let dsq = dx * dx + dy * dy
        if (dsq < 1) {
          const angle = ((hashNumber(`${sa.id}:${sb.id}`) % 360) * Math.PI) / 180
          dx = Math.cos(angle); dy = Math.sin(angle); dsq = 1
        }
        const f = repulsion / Math.max(dsq, 900)
        const fx = (dx / Math.sqrt(dsq)) * f
        const fy = (dy / Math.sqrt(dsq)) * f
        forces.get(sa.id).x -= fx; forces.get(sa.id).y -= fy
        forces.get(sb.id).x += fx; forces.get(sb.id).y += fy
      }
    }

    for (const e of graphEdges) {
      const dx = e.target.x - e.source.x
      const dy = e.target.y - e.source.y
      const dist = Math.max(1, Math.hypot(dx, dy))
      const w = clamp(Number(e.weight) || 0.5, 0.2, 1.4)
      const f = (dist - springLen) * springK * w
      const fx = (dx / dist) * f, fy = (dy / dist) * f
      forces.get(e.source.id).x += fx; forces.get(e.source.id).y += fy
      forces.get(e.target.id).x -= fx; forces.get(e.target.id).y -= fy
    }

    for (const node of laidOut) {
      const f = forces.get(node.id)
      const anchor = typeAnchor(node, cx, cy, spreadX, spreadY)
      f.x += (anchor.x - node.x) * anchorK
      f.y += (anchor.y - node.y) * anchorK
      f.x += (cx - node.x) * centerK
      f.y += (cy - node.y) * centerK

      if (pinned.has(node.id)) continue
      node.vx = (node.vx + f.x) * 0.72
      node.vy = (node.vy + f.y) * 0.72
      node.x = clamp(node.x + node.vx, minX, maxX)
      node.y = clamp(node.y + node.vy, minY, maxY)
    }
  }

  for (const node of laidOut) { delete node.vx; delete node.vy }
  return laidOut
}
