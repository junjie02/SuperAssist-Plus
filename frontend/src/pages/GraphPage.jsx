import { useState, useEffect, useCallback } from 'react'
import { api } from '../lib/api'
import GraphCanvas from '../components/GraphCanvas'

export default function GraphPage() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('graph') // 'graph' | 'list'
  const [filter, setFilter] = useState('all')

  const load = useCallback(() => {
    api.get('/graph').then(setData).catch(e => setError(e.message))
  }, [])

  useEffect(() => { load() }, [load])

  if (error) return <div className="graph-page"><p className="error">Failed to load graph: {error}</p></div>
  if (!data) return <div className="graph-page"><p>Loading graph...</p></div>

  const { stats, nodes, edges } = data

  // Apply client-side filter
  const filteredNodes = filter === 'all' ? nodes : nodes.filter(n => n.type === filter)
  const filteredNodeIds = new Set(filteredNodes.map(n => n.id))
  const filteredEdges = edges.filter(e => filteredNodeIds.has(e.source_id) && filteredNodeIds.has(e.target_id))

  return (
    <div className="graph-page">
      {/* ---- Tabs ---- */}
      <div className="graph-tabs">
        <button className={`graph-tab ${tab === 'graph' ? 'active' : ''}`} onClick={() => setTab('graph')}>
          🧠 Graph
        </button>
        <button className={`graph-tab ${tab === 'list' ? 'active' : ''}`} onClick={() => setTab('list')}>
          📋 List
        </button>
        <button className="graph-refresh-btn" onClick={load}>↻ Refresh</button>
        <span className="graph-updated">Loaded {new Date().toLocaleTimeString()}</span>
      </div>

      {/* ---- Stats ---- */}
      <div className="graph-stats">
        <Stat label="Nodes" value={stats.nodes} />
        <Stat label="Edges" value={stats.edges} />
        <Stat label="Concepts" value={stats.by_type?.concept || 0} />
        <Stat label="Intents" value={stats.by_type?.intent || 0} />
        <Stat label="Events" value={stats.by_type?.event || 0} />
        <Stat label="Time" value={stats.by_type?.time || 0} />
      </div>

      {/* ---- Filter (shared) ---- */}
      <div className="graph-filter">
        <label>Filter: </label>
        <select value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="all">All</option>
          <option value="event">Events</option>
          <option value="concept">Concepts</option>
          <option value="intent">Intents</option>
          <option value="time">Time</option>
        </select>
      </div>

      {/* ---- Content ---- */}
      {tab === 'graph' ? (
        <div className="graph-canvas-container">
          {filteredNodes.length === 0 ? (
            <div className="empty-graph">No nodes match the filter.</div>
          ) : (
            <GraphCanvas nodes={filteredNodes} edges={filteredEdges} />
          )}
        </div>
      ) : (
        <div className="graph-lists">
          <NodeTable nodes={filteredNodes} />
          <EdgeTable edges={filteredEdges} />
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }) {
  return <div className="stat-tile"><span className="stat-value">{value}</span><span className="stat-label">{label}</span></div>
}

function NodeTable({ nodes }) {
  const hasRecall = nodes.some(n => n.active_recall)
  return (
    <div className="graph-table-wrap">
      <h3>Nodes ({nodes.length})</h3>
      <table>
        <thead><tr><th>Type</th><th>Title</th><th>Score</th><th>Access</th>{hasRecall && <th>Recall</th>}</tr></thead>
        <tbody>
          {nodes.slice(0, 200).map(n => (
            <tr key={n.id} className={n.active_recall ? 'recall-active' : ''}>
              <td><span className={`type-badge ${n.type}`}>{n.type}</span></td>
              <td>{n.title}</td>
              <td>{(n.importance || 0).toFixed(2)}</td>
              <td>{n.access_count || 0}</td>
              {hasRecall && <td>{n.active_recall ? `${n.recall_score?.toFixed(2)} (${n.recall_tier})` : '-'}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function EdgeTable({ edges }) {
  return (
    <div className="graph-table-wrap">
      <h3>Edges ({edges.length})</h3>
      <table>
        <thead><tr><th>Type</th><th>Source</th><th>Target</th><th>Weight</th></tr></thead>
        <tbody>
          {edges.slice(0, 200).map(e => (
            <tr key={e.id}>
              <td><span className="type-badge edge">{e.edge_type}</span></td>
              <td className="mono">{e.source_id?.slice(0, 20)}</td>
              <td className="mono">{e.target_id?.slice(0, 20)}</td>
              <td>{e.weight?.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
