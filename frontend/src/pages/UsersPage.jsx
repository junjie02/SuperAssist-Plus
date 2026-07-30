import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  ArrowLeft,
  Bot,
  GitBranch,
  LoaderCircle,
  Maximize2,
  MessageSquare,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  UserRound,
  Users,
  X,
} from 'lucide-react'
import GraphCanvas from '../components/GraphCanvas'
import { useAuth } from '../hooks/useAuth'
import { api } from '../lib/api'

export default function UsersPage() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState([])
  const [threads, setThreads] = useState([])
  const [history, setHistory] = useState([])
  const [graph, setGraph] = useState(null)
  const [selectedUserId, setSelectedUserId] = useState('')
  const [selectedThreadId, setSelectedThreadId] = useState('')
  const [query, setQuery] = useState('')
  const [pane, setPane] = useState('users')
  const [loadingUsers, setLoadingUsers] = useState(true)
  const [loadingThreads, setLoadingThreads] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [loadingGraph, setLoadingGraph] = useState(false)
  const [graphError, setGraphError] = useState('')
  const [graphExpanded, setGraphExpanded] = useState(false)
  const [deletingMessage, setDeletingMessage] = useState(null)
  const [error, setError] = useState('')

  const selectedUser = users.find(item => item.id === selectedUserId) || null
  const selectedThread = threads.find(item => item.thread_id === selectedThreadId) || null
  const filteredUsers = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return users
    return users.filter(item => (
      item.username.toLowerCase().includes(normalized)
      || item.id.toLowerCase().includes(normalized)
      || item.channel.toLowerCase().includes(normalized)
    ))
  }, [query, users])

  const loadUsers = useCallback(async () => {
    setLoadingUsers(true)
    setError('')
    try {
      const data = await api.get('/admin/users')
      setUsers(data)
      setSelectedUserId(current => (
        data.some(item => item.id === current) ? current : (data[0]?.id || '')
      ))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingUsers(false)
    }
  }, [])

  useEffect(() => {
    if (currentUser?.isAdmin) loadUsers()
  }, [currentUser?.isAdmin, loadUsers])

  useEffect(() => {
    if (!selectedUserId) {
      setThreads([])
      setSelectedThreadId('')
      return
    }
    let cancelled = false
    setLoadingThreads(true)
    setHistory([])
    api.get(`/admin/users/${encodeURIComponent(selectedUserId)}/threads`)
      .then(data => {
        if (cancelled) return
        setThreads(data)
        setSelectedThreadId(data[0]?.thread_id || '')
      })
      .catch(e => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoadingThreads(false) })
    return () => { cancelled = true }
  }, [selectedUserId])

  useEffect(() => {
    if (!selectedUserId || !selectedThreadId) {
      setHistory([])
      return
    }
    let cancelled = false
    setLoadingHistory(true)
    api.get(`/admin/users/${encodeURIComponent(selectedUserId)}/threads/${encodeURIComponent(selectedThreadId)}/history`)
      .then(data => { if (!cancelled) setHistory(data) })
      .catch(e => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoadingHistory(false) })
    return () => { cancelled = true }
  }, [selectedUserId, selectedThreadId])

  const loadGraph = useCallback(async () => {
    if (!selectedUserId) {
      setGraph(null)
      return
    }
    setLoadingGraph(true)
    setGraphError('')
    try {
      setGraph(await api.get(`/admin/users/${encodeURIComponent(selectedUserId)}/graph`))
    } catch (e) {
      setGraph(null)
      setGraphError(e.message)
    } finally {
      setLoadingGraph(false)
    }
  }, [selectedUserId])

  useEffect(() => { loadGraph() }, [loadGraph])

  useEffect(() => {
    if (!graphExpanded) return undefined
    const closeOnEscape = event => {
      if (event.key === 'Escape') setGraphExpanded(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [graphExpanded])

  useEffect(() => { setGraphExpanded(false) }, [selectedUserId])

  const deleteMessage = useCallback(async message => {
    if (!selectedUserId || !selectedThreadId) return
    const confirmed = window.confirm(
      'Delete this short-memory record? This does not remove compressed summaries or long-term graph nodes.'
    )
    if (!confirmed) return
    setDeletingMessage(message.record_index)
    setError('')
    try {
      await api.del(`/admin/users/${encodeURIComponent(selectedUserId)}/threads/${encodeURIComponent(selectedThreadId)}/messages/${message.record_index}`)
      const nextHistory = await api.get(`/admin/users/${encodeURIComponent(selectedUserId)}/threads/${encodeURIComponent(selectedThreadId)}/history`)
      setHistory(nextHistory)
      const nextThreads = await api.get(`/admin/users/${encodeURIComponent(selectedUserId)}/threads`)
      setThreads(nextThreads)
      await loadUsers()
    } catch (e) {
      setError(e.message)
    } finally {
      setDeletingMessage(null)
    }
  }, [loadUsers, selectedThreadId, selectedUserId])

  if (!currentUser?.isAdmin) {
    return (
      <div className="admin-access-denied">
        <ShieldCheck size={28} />
        <h1>Administrator access required</h1>
      </div>
    )
  }

  return (
    <div className="users-page">
      <header className="users-header">
        <div>
          <h1>Users</h1>
          <span>{users.length} identities across all channels</span>
        </div>
        <button className="icon-btn" onClick={loadUsers} disabled={loadingUsers} title="Refresh users">
          <RefreshCw size={18} className={loadingUsers ? 'spin' : ''} />
        </button>
      </header>

      <div className="admin-mobile-tabs" aria-label="User record views">
        <button className={pane === 'users' ? 'active' : ''} onClick={() => setPane('users')}><Users size={16} /> Users</button>
        <button className={pane === 'threads' ? 'active' : ''} onClick={() => setPane('threads')} disabled={!selectedUser}><MessageSquare size={16} /> Chats</button>
        <button className={pane === 'history' ? 'active' : ''} onClick={() => setPane('history')} disabled={!selectedThread}><Bot size={16} /> Messages</button>
        <button className={pane === 'graph' ? 'active' : ''} onClick={() => setPane('graph')} disabled={!selectedUser}><GitBranch size={16} /> Graph</button>
      </div>

      {error && <div className="users-error">{error}</div>}

      <div className="users-workspace">
        <section className={`user-directory ${pane === 'users' ? 'mobile-active' : ''}`}>
          <div className="directory-search">
            <Search size={16} />
            <input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search users" />
          </div>
          <div className="directory-list">
            {loadingUsers ? <LoadingState label="Loading users" /> : filteredUsers.map(item => (
              <button
                key={item.id}
                className={`directory-row ${selectedUserId === item.id ? 'active' : ''}`}
                onClick={() => { setSelectedUserId(item.id); setPane('threads') }}
              >
                <ChannelIcon channel={item.channel} />
                <span className="directory-main">
                  <strong>{item.username}</strong>
                  <small>{item.id}</small>
                </span>
                <span className="directory-stats">
                  {item.is_admin && <ShieldCheck size={14} aria-label="Administrator" />}
                  <b>{item.conversation_count}</b>
                </span>
              </button>
            ))}
            {!loadingUsers && filteredUsers.length === 0 && <EmptyState label="No users found" />}
          </div>
        </section>

        <section className={`user-threads ${pane === 'threads' ? 'mobile-active' : ''}`}>
          <div className="records-heading">
            <button className="mobile-back" onClick={() => setPane('users')} title="Back to users"><ArrowLeft size={17} /></button>
            <div>
              <h2>{selectedUser?.username || 'Select a user'}</h2>
              <span>{selectedUser ? `${channelLabel(selectedUser.channel)} · ${selectedUser.message_count} messages` : 'Conversation history'}</span>
            </div>
          </div>
          <div className="thread-record-list">
            {loadingThreads ? <LoadingState label="Loading conversations" /> : threads.map(thread => (
              <button
                key={thread.thread_id}
                className={`thread-record-row ${selectedThreadId === thread.thread_id ? 'active' : ''}`}
                onClick={() => { setSelectedThreadId(thread.thread_id); setPane('history') }}
              >
                <span className="thread-record-preview">{thread.preview || '(empty)'}</span>
                <span className="thread-record-meta">
                  <time>{formatDate(thread.updated_at)}</time>
                  <b>{thread.message_count}</b>
                </span>
              </button>
            ))}
            {!loadingThreads && selectedUser && threads.length === 0 && <EmptyState label="No conversations" />}
          </div>
        </section>

        <div className={`user-detail ${pane === 'history' || pane === 'graph' ? 'mobile-active' : ''}`}>
          <section className={`user-transcript ${pane === 'history' ? 'mobile-active' : ''}`}>
            <div className="records-heading transcript-heading">
              <button className="mobile-back" onClick={() => setPane('threads')} title="Back to conversations"><ArrowLeft size={17} /></button>
              <div>
                <h2>{selectedThread?.preview || 'Conversation'}</h2>
                <span>{selectedThread ? `${selectedThread.message_count} messages · ${formatDate(selectedThread.updated_at)}` : 'Select a conversation'}</span>
              </div>
            </div>
            <div className="transcript-list">
              {loadingHistory ? <LoadingState label="Loading messages" /> : history.map(message => (
                <article className={`transcript-message ${message.role}`} key={message.record_index}>
                  <header>
                    <span>{roleLabel(message.role)}</span>
                    <div className="transcript-message-actions">
                      <time>{formatMessageTime(message.created_at)}</time>
                      <button
                        className="message-delete-btn"
                        onClick={() => deleteMessage(message)}
                        disabled={deletingMessage !== null}
                        title="Delete short-memory record"
                        aria-label="Delete short-memory record"
                      >
                        {deletingMessage === message.record_index ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}
                      </button>
                    </div>
                  </header>
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || ''}</ReactMarkdown>
                  </div>
                </article>
              ))}
              {!loadingHistory && selectedThread && history.length === 0 && <EmptyState label="No messages" />}
              {!loadingHistory && !selectedThread && <EmptyState label="Select a conversation to inspect" />}
            </div>
          </section>

          <section className={`user-graph ${pane === 'graph' ? 'mobile-active' : ''}`}>
            <div className="records-heading user-graph-heading">
              <button className="mobile-back" onClick={() => setPane('threads')} title="Back to conversations"><ArrowLeft size={17} /></button>
              <GitBranch size={16} />
              <div>
                <h2>Memory Graph</h2>
                <span>{graph ? `${graph.stats?.nodes || 0} nodes · ${graph.stats?.edges || 0} edges` : selectedUser?.username || 'Select a user'}</span>
              </div>
              <div className="graph-heading-actions">
                <button
                  className="icon-btn"
                  onClick={() => setGraphExpanded(true)}
                  disabled={!selectedUser}
                  title="Enlarge graph"
                  aria-label="Enlarge graph"
                >
                  <Maximize2 size={15} />
                </button>
                <button className="icon-btn" onClick={loadGraph} disabled={loadingGraph || !selectedUser} title="Refresh graph" aria-label="Refresh graph">
                  <RefreshCw size={15} className={loadingGraph ? 'spin' : ''} />
                </button>
              </div>
            </div>
            <div className="user-graph-canvas">
              {loadingGraph && !graph ? <LoadingState label="Loading memory graph" /> : graphError ? (
                <div className="records-state graph-error">Failed to load graph: {graphError}</div>
              ) : (
                <GraphCanvas
                  nodes={graph?.nodes || []}
                  edges={graph?.edges || []}
                  emptyMessage="This user has no memory nodes yet."
                  showEdgeLabels={false}
                />
              )}
            </div>
          </section>
        </div>
      </div>

      {graphExpanded && (
        <div
          className="graph-expanded-overlay"
          onMouseDown={event => {
            if (event.target === event.currentTarget) setGraphExpanded(false)
          }}
        >
          <section className="graph-expanded-dialog" role="dialog" aria-modal="true" aria-label={`${selectedUser?.username || 'User'} memory graph`}>
            <header className="graph-expanded-heading">
              <div className="graph-expanded-title">
                <GitBranch size={18} />
                <div>
                  <h2>{selectedUser?.username || 'User'} Memory Graph</h2>
                  <span>{graph ? `${graph.stats?.nodes || 0} nodes · ${graph.stats?.edges || 0} edges` : 'Memory graph'}</span>
                </div>
              </div>
              <div className="graph-heading-actions">
                <button className="icon-btn" onClick={loadGraph} disabled={loadingGraph} title="Refresh graph" aria-label="Refresh graph">
                  <RefreshCw size={16} className={loadingGraph ? 'spin' : ''} />
                </button>
                <button className="icon-btn" onClick={() => setGraphExpanded(false)} title="Close enlarged graph" aria-label="Close enlarged graph">
                  <X size={17} />
                </button>
              </div>
            </header>
            <div className="graph-expanded-canvas">
              {loadingGraph && !graph ? <LoadingState label="Loading memory graph" /> : graphError ? (
                <div className="records-state graph-error">Failed to load graph: {graphError}</div>
              ) : (
                <GraphCanvas
                  nodes={graph?.nodes || []}
                  edges={graph?.edges || []}
                  emptyMessage="This user has no memory nodes yet."
                  showEdgeLabels={false}
                />
              )}
            </div>
          </section>
        </div>
      )}
    </div>
  )
}

function ChannelIcon({ channel }) {
  return (
    <span className={`channel-icon ${channel}`} title={channelLabel(channel)}>
      {channel === 'web' ? <UserRound size={17} /> : channel.startsWith('feishu') ? 'F' : 'W'}
    </span>
  )
}

function LoadingState({ label }) {
  return <div className="records-state"><LoaderCircle className="spin" size={18} /> {label}</div>
}

function EmptyState({ label }) {
  return <div className="records-state">{label}</div>
}

function channelLabel(channel) {
  const labels = { web: 'Web', feishu: 'Feishu', 'feishu-group': 'Feishu group', wecom: 'WeCom', 'wecom-group': 'WeCom group', 'wecom-rpa': 'WeCom RPA' }
  return labels[channel] || channel
}

function roleLabel(role) {
  if (role === 'user') return 'User'
  if (role === 'assistant') return 'Assistant'
  if (role === 'tool_event') return 'Tool event'
  return role || 'System'
}

function formatDate(value) {
  if (!value) return ''
  return new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function formatMessageTime(value) {
  if (!value) return ''
  return new Date(value).toLocaleString()
}
