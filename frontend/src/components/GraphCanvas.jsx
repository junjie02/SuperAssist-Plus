import { useRef, useEffect, useState, useCallback } from 'react'
import { layoutNodes, edgeGeometry, edgeCurveOffsets, displayScore, shortText, TYPE_COLORS } from '../lib/graph-layout'

const SVG_NS = 'http://www.w3.org/2000/svg'
const MIN_ZOOM = 0.35; const MAX_ZOOM = 3

export default function GraphCanvas({ nodes, edges }) {
  const svgRef = useRef(null)
  const [selected, setSelected] = useState(null)
  const posRef = useRef(new Map())
  const pinnedRef = useRef(new Set())
  const stRef = useRef({ laidOut: [], vp: { x: 0, y: 0, scale: 1 } })

  // Rebuild layout only when data changes (NOT on selection)
  useEffect(() => {
    const svg = svgRef.current
    if (!svg || !nodes.length) return
    const w = Math.max(svg.clientWidth, 720)
    const h = Math.max(svg.clientHeight, 560)

    const currentIds = new Set(nodes.map(n => n.id))
    for (const id of posRef.current.keys()) { if (!currentIds.has(id)) posRef.current.delete(id) }
    for (const id of pinnedRef.current) { if (!currentIds.has(id)) pinnedRef.current.delete(id) }

    const laidOut = layoutNodes(nodes, edges, w, h, { savedPositions: posRef.current, pinned: pinnedRef.current })
    const es = buildEdgeState(edges, laidOut)
    stRef.current = { ...stRef.current, laidOut, edgeState: es }
    fullRender(svg, laidOut, edges, w, h)
  }, [nodes, edges])

  // Update selection highlight without re-layout
  useEffect(() => {
    const selId = selected?.id || ''
    const svg = svgRef.current
    if (!svg) return
    svg.querySelectorAll('.node').forEach(g => {
      g.classList.toggle('selected', g.dataset.id === selId)
    })
    svg.querySelectorAll('.edge').forEach(g => {
      const isSel = g.dataset.id === selId
      const isConn = selId && (g.dataset.sourceId === selId || g.dataset.targetId === selId)
      g.classList.toggle('selected', isSel)
      g.classList.toggle('connected', isConn)
    })
  }, [selected])

  // ---- Interaction helpers ----

  const toSvgPt = useCallback((e) => {
    const svg = svgRef.current
    const pt = svg.createSVGPoint()
    pt.x = e.clientX; pt.y = e.clientY
    const m = svg.getScreenCTM()
    return m ? pt.matrixTransform(m.inverse()) : { x: e.offsetX, y: e.offsetY }
  }, [])

  const toGraphPt = useCallback((e) => {
    const p = toSvgPt(e)
    const v = stRef.current.vp
    return { x: (p.x - v.x) / v.scale, y: (p.y - v.y) / v.scale }
  }, [toSvgPt])

  const applyVp = useCallback(() => {
    const g = svgRef.current?.querySelector('.viewport-layer')
    if (g) { const v = stRef.current.vp; g.setAttribute('transform', `translate(${v.x}, ${v.y}) scale(${v.scale})`) }
  }, [])

  // ---- Drag & pan ----

  const onDown = useCallback((e) => {
    if (e.button !== 0) return
    const svg = svgRef.current
    const nodeEl = e.target.closest('.node')
    if (nodeEl) {
      e.preventDefault(); e.stopPropagation()
      const n = stRef.current.laidOut.find(x => x.id === nodeEl.dataset.id)
      if (!n) return
      const gp = toGraphPt(e)  // graph-space pointer position
      stRef.current.drag = {
        id: n.id,
        px: gp.x, py: gp.y,    // pointer start in graph space
        nx: n.x, ny: n.y,      // node start in graph space
        moved: false,
      }
      nodeEl.classList.add('dragging')
      svg.setPointerCapture(e.pointerId)
      return
    }
    if (e.target.closest('.edge')) return
    // Canvas pan
    const pt = toSvgPt(e)
    const v = stRef.current.vp
    stRef.current.pan = { sx: pt.x, sy: pt.y, vx: v.x, vy: v.y, moved: false }
    svg.classList.add('is-panning')
    svg.setPointerCapture(e.pointerId)
  }, [toSvgPt, toGraphPt])

  const onMove = useCallback((e) => {
    const st = stRef.current
    if (st.drag) {
      const gp = toGraphPt(e)
      const n = st.laidOut.find(x => x.id === st.drag.id)
      if (!n) return
      const dx = gp.x - st.drag.px
      const dy = gp.y - st.drag.py
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) st.drag.moved = true
      n.x = st.drag.nx + dx
      n.y = st.drag.ny + dy
      posRef.current.set(n.id, { x: n.x, y: n.y })
      if (st.drag.moved) pinnedRef.current.add(n.id)
      updatePositions(st)
      return
    }
    if (st.pan) {
      const pt = toSvgPt(e)
      const dx = pt.x - st.pan.sx, dy = pt.y - st.pan.sy
      if (Math.abs(dx) > 1 || Math.abs(dy) > 1) st.pan.moved = true
      st.vp.x = st.pan.vx + dx; st.vp.y = st.pan.vy + dy
      applyVp()
    }
  }, [toSvgPt, toGraphPt, applyVp])

  const onUp = useCallback((e) => {
    const st = stRef.current
    const svg = svgRef.current
    if (st.drag) {
      svg.querySelector('.dragging')?.classList.remove('dragging')
      if (!st.drag.moved) {
        const n = st.laidOut.find(x => x.id === st.drag.id)
        if (n) setSelected({ kind: 'node', ...n })
      }
      st.drag = null
    }
    if (st.pan) { st.pan = null; svg.classList.remove('is-panning') }
    try { svg.releasePointerCapture(e.pointerId) } catch { /* ok */ }
  }, [])

  useEffect(() => {
    const svg = svgRef.current; if (!svg) return
    const wh = (e) => { e.preventDefault(); zoom(e, stRef, toSvgPt, applyVp) }
    svg.addEventListener('wheel', wh, { passive: false })
    return () => svg.removeEventListener('wheel', wh)
  }, [toSvgPt, applyVp])

  return (
    <div style={{ position: 'relative', flex: 1, overflow: 'hidden' }}>
      <svg ref={svgRef} style={{ width: '100%', height: '100%', cursor: 'grab', touchAction: 'none' }}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} />
      {nodes.length === 0 && (
        <div className="empty-graph">No nodes yet. Send a message to start building the memory graph.</div>
      )}
      {selected && (
        <div className="graph-detail-panel">
          <button className="detail-close" onClick={() => setSelected(null)}>✕</button>
          <h4>{selected.kind === 'node' ? selected.title : `${selected.edge_type} · ${selected.weight?.toFixed(2)}`}</h4>
          <p>{selected.kind === 'node'
            ? `${selected.type} · score ${displayScore(selected).toFixed(2)} · ${selected.description || ''}`
            : `${selected.sourceTitle || selected.source_id} → ${selected.targetTitle || selected.target_id}`}</p>
          {selected.recall_components && (
            <div className="detail-scores" style={{ display: 'flex', gap: 10, fontSize: '0.78rem', color: 'var(--muted)', marginTop: 4 }}>
              <span>PR {selected.recall_components.pagerank?.toFixed(2)}</span>
              <span>rec {selected.recall_components.recency?.toFixed(2)}</span>
              <span>acc {selected.recall_components.access?.toFixed(2)}</span>
              <span>urg {selected.recall_components.urgency?.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---- Zoom ----

function zoom(e, stRef, toSvgPt, applyVp) {
  const st = stRef.current; if (!st.laidOut?.length) return
  const pt = toSvgPt(e); const v = st.vp
  const gx = (pt.x - v.x) / v.scale; const gy = (pt.y - v.y) / v.scale
  const zf = e.deltaY < 0 ? 1.12 : 1 / 1.12
  st.vp.scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.scale * zf))
  st.vp.x = pt.x - gx * st.vp.scale; st.vp.y = pt.y - gy * st.vp.scale
  applyVp()
}

// ---- Rendering ----

function fullRender(svg, laidOut, edges, w, h) {
  const nodeMap = new Map(laidOut.map(n => [n.id, n]))
  const offsets = edgeCurveOffsets(edges)
  const validEdges = edges.filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))

  svg.setAttribute('viewBox', `0 0 ${w} ${h}`)
  svg.innerHTML = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>`

  const vp = mk('g', { class: 'viewport-layer' })
  const elL = mk('g', { class: 'edge-layer' })
  const nlL = mk('g', { class: 'node-layer' })

  for (const e of validEdges) {
    const s = nodeMap.get(e.source_id), t = nodeMap.get(e.target_id)
    const off = offsets.get(e.id) || 0; const geo = edgeGeometry(s, t, off)
    const g = mk('g', { class: 'edge', 'data-id': e.id, 'data-source-id': e.source_id, 'data-target-id': e.target_id, 'data-curve-offset': String(off) })
    g.append(
      mk('path', { class: 'edge-line', d: geo.path, fill: 'none', 'stroke-width': String(1.1 + e.weight * 2.3) }),
      txt('text', { class: 'edge-label', x: String(geo.labelX), y: String(geo.labelY), 'text-anchor': 'middle' }, e.edge_type || '')
    )
    g.addEventListener('click', () => {
      // use a custom event or direct callback
      const detail = { kind: 'edge', ...e, sourceTitle: s.title, targetTitle: t.title }
      window.__graphSelect?.(detail)
    })
    elL.append(g)
  }

  for (const n of laidOut) {
    const g = mk('g', { class: `node${n.active_recall ? ' active-recall' : ''}${n.recall_tier ? ` tier-${n.recall_tier}` : ''}`, 'data-id': n.id })
    g.setAttribute('transform', `translate(${n.x}, ${n.y})`)
    g.append(
      mk('circle', { class: n.type, r: String(18 + Math.min(12, displayScore(n) * 12)), fill: TYPE_COLORS[n.type] || '#64748b' }),
      txt('text', { y: '38', 'text-anchor': 'middle' }, shortText(n.title, 28))
    )
    nlL.append(g)
  }

  vp.append(elL, nlL); svg.append(vp)
}

function updatePositions(st) {
  const es = st.edgeState; if (!es) return
  document.querySelectorAll('.node-layer .node').forEach(g => {
    const n = st.laidOut.find(x => x.id === g.dataset.id)
    if (n) g.setAttribute('transform', `translate(${n.x}, ${n.y})`)
  })
  document.querySelectorAll('.edge-layer .edge').forEach(g => {
    const s = es.nodeMap.get(g.dataset.sourceId), t = es.nodeMap.get(g.dataset.targetId)
    if (!s || !t) return
    const geo = edgeGeometry(s, t, Number(g.dataset.curveOffset) || 0)
    const p = g.querySelector('.edge-line'); if (p) p.setAttribute('d', geo.path)
    const l = g.querySelector('.edge-label'); if (l) { l.setAttribute('x', String(geo.labelX)); l.setAttribute('y', String(geo.labelY)) }
  })
}

function buildEdgeState(edges, laidOut) {
  const nodeMap = new Map(laidOut.map(n => [n.id, n]))
  const valid = edges.filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))
  return { edges: valid, curveOffsets: edgeCurveOffsets(valid), nodeMap }
}

function mk(name, attrs) {
  const e = document.createElementNS(SVG_NS, name)
  for (const [k, v] of Object.entries(attrs)) { if (v != null) e.setAttribute(k, String(v)) }
  return e
}

function txt(name, attrs, text) {
  const e = mk(name, attrs); e.textContent = text; return e
}
