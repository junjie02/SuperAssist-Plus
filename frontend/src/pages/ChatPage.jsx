import { useState, useRef, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import useWebSocket from '../hooks/useWebSocket'
import { api } from '../lib/api'

const THINKING_DOTS = ['  Thinking .', '  Thinking ..', '  Thinking ...']

export default function ChatPage() {
  const [messages, setMessages] = useState([])
  const [currentAI, setCurrentAI] = useState('')
  const [thinking, setThinking] = useState(false)
  const [thinkingDots, setThinkingDots] = useState(0)
  const [threadId, setThreadId] = useState(null)
  const [threads, setThreads] = useState([])
  const [toolCalls, setToolCalls] = useState([])
  const messagesEnd = useRef(null)
  const { send, on, connect, ready } = useWebSocket()
  const pendingRef = useRef(null)
  const thinkingTimer = useRef(null)
  const loadedThreadRef = useRef(null)   // track which thread's history is loaded

  useEffect(() => {
    if (thinking) {
      thinkingTimer.current = setInterval(() => {
        setThinkingDots(d => (d + 1) % THINKING_DOTS.length)
      }, 400)
    } else {
      if (thinkingTimer.current) clearInterval(thinkingTimer.current)
      setThinkingDots(0)
    }
    return () => { if (thinkingTimer.current) clearInterval(thinkingTimer.current) }
  }, [thinking])

  useEffect(() => { api.get('/threads').then(setThreads).catch(() => {}) }, [])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAI, thinking])

  useEffect(() => {
    on('agent_text', (data) => { setThinking(false); setCurrentAI(data.content || '') })

    on('subagent_text', (data) => {
      setThinking(false)
      const desc = data.metadata?.description || 'subagent'
      setCurrentAI(prev => prev + `\n\n[Subagent: ${desc}]\n${data.content || ''}`)
    })

    on('thinking', () => { setThinking(true) })

    on('tool_start', (data) => {
      setThinking(false)
      setToolCalls(prev => [...prev, { tool: data.tool || data.name, args: data.args, status: 'running' }])
    })

    on('tool_result', (data) => {
      setToolCalls(prev => prev.map(tc =>
        (tc.tool === (data.tool || data.name) && tc.status === 'running')
          ? { ...tc, status: data.status || 'done', result: data.content, error: data.error }
          : tc
      ))
    })

    on('done', (data) => {
      setThinking(false)
      const answer = data.answer || data.content || currentAI || ''
      const tid = data.thread_id || threadId
      setMessages(prev => [...prev, { role: 'assistant', content: answer, thread_id: tid, toolCalls: [...toolCalls] }])
      setCurrentAI('')
      setToolCalls([])
      if (tid && tid !== threadId) {
        setThreadId(tid)
        loadedThreadRef.current = tid
        api.get('/threads').then(setThreads).catch(() => {})
      }
    })

    on('error', (data) => {
      setThinking(false)
      setMessages(prev => [...prev, { role: 'system', content: `❌ ${data.message || data.content || 'Unknown error'}` }])
      setCurrentAI('')
      setToolCalls([])
    })

    on('*', () => {})
  }, [on, toolCalls, threadId, currentAI])

  const doSend = useCallback((text, tid) => {
    if (!text.trim()) return
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setCurrentAI('')
    setToolCalls([])
    setThinking(true)
    const ok = send(text, tid)
    if (!ok) { pendingRef.current = { text, tid }; connect() }
  }, [send, connect])

  useEffect(() => {
    if (ready && pendingRef.current) {
      const { text, tid } = pendingRef.current
      pendingRef.current = null
      send(text, tid)
      setThinking(true)
    }
  }, [ready, send])

  const handleNewThread = useCallback(() => {
    setThreadId(null)
    setMessages([])
    setCurrentAI('')
    setToolCalls([])
    setThinking(false)
    loadedThreadRef.current = null
  }, [])

  const handleThreadSelect = useCallback(async (tid) => {
    if (tid === loadedThreadRef.current) return
    setThreadId(tid)
    setCurrentAI('')
    setToolCalls([])
    setThinking(false)
    try {
      const history = await api.get(`/threads/${tid}/history`)
      setMessages(history.map(h => ({
        role: h.role === 'tool_event' ? 'tool' : h.role,
        content: h.content || '',
      })))
      loadedThreadRef.current = tid
    } catch {
      setMessages([])
      loadedThreadRef.current = tid
    }
  }, [])

  const handleDeleteThread = useCallback(async (tid) => {
    await api.del(`/threads/${tid}`).catch(() => {})
    if (threadId === tid || loadedThreadRef.current === tid) {
      handleNewThread()
    }
    api.get('/threads').then(setThreads).catch(() => {})
  }, [threadId, handleNewThread])

  return (
    <div className="chat-page">
      {/* ---- Main chat area ---- */}
      <div className="chat-main">
        <div className="message-list">
          {messages.length === 0 && !currentAI && !thinking && (
            <div className="empty-chat">
              <h2>{'🧠'} SuperAssist</h2>
              <p>Ask me anything. I have long-term memory and can use tools.</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {(currentAI || toolCalls.length > 0) && (
            <MessageBubble message={{ role: 'assistant', content: currentAI, toolCalls }} streaming />
          )}

          {thinking && !currentAI && toolCalls.length === 0 && (
            <div className="message assistant">
              <div className="message-role">AI</div>
              <div className="message-content">
                <p className="thinking-text">{THINKING_DOTS[thinkingDots]}</p>
              </div>
            </div>
          )}

          <div ref={messagesEnd} />
        </div>

        <ChatInput onSend={(text) => doSend(text, threadId)} disabled={thinking} />
      </div>

      {/* ---- Thread list panel ---- */}
      <aside className="thread-panel">
        <div className="thread-panel-header">
          <h3>Conversations</h3>
          <button onClick={handleNewThread} className="new-thread-btn">+ New</button>
        </div>
        <div className="thread-list">
          {threads.length === 0 && (
            <p className="thread-empty">No conversations yet.</p>
          )}
          {threads.map(t => (
            <div
              key={t.thread_id}
              className={`thread-item ${(threadId === t.thread_id || loadedThreadRef.current === t.thread_id) ? 'active' : ''}`}
              onClick={() => handleThreadSelect(t.thread_id)}
            >
              <div className="thread-item-main">
                <span className="thread-preview">{t.preview || '(empty)'}</span>
                <span className="thread-time">{fmtTime(t.updated_at)}</span>
              </div>
              <button
                className="thread-delete"
                onClick={(e) => { e.stopPropagation(); handleDeleteThread(t.thread_id) }}
                title="Delete"
              >
                {'✕'}
              </button>
            </div>
          ))}
        </div>
      </aside>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MessageBubble({ message, streaming }) {
  const { role, content, toolCalls: tc } = message
  const isUser = role === 'user'
  const isSystem = role === 'system'

  return (
    <div className={`message ${isUser ? 'user' : isSystem ? 'system' : 'assistant'}`}>
      <div className="message-role">{isUser ? 'You' : isSystem ? 'System' : 'AI'}</div>
      <div className="message-content">
        {content && (
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            {streaming && <span className="cursor">{'▌'}</span>}
          </div>
        )}
        {tc && tc.length > 0 && (
          <div className="tool-calls">
            {tc.map((t, i) => <ToolCallCard key={i} tool={t} />)}
          </div>
        )}
      </div>
    </div>
  )
}

function toolSummary(tool) {
  const args = tool.args || {}
  switch (tool.tool) {
    case 'web_search': return `搜索: ${args.query || ''}`
    case 'web_fetch': return `抓取: ${(args.url || '').slice(0, 50)}`
    case 'read_file': return `读取: ${(args.path || '').split('/').pop() || args.path || ''}`
    case 'write_file': return `写入: ${(args.path || '').split('/').pop() || args.path || ''}`
    case 'shell': return `执行: ${(args.command || '').slice(0, 40)}`
    case 'task': return `${args.description || '子任务'}`
    default: return ''
  }
}

function ToolCallCard({ tool }) {
  const [expanded, setExpanded] = useState(false)
  const isRunning = tool.status === 'running'
  const isError = tool.status === 'error'
  const summary = toolSummary(tool)

  return (
    <div className={`tool-card ${isRunning ? 'running' : isError ? 'error' : 'done'}`}>
      <button className="tool-card-header" onClick={() => setExpanded(!expanded)}>
        <span className="tool-card-title">
          <span className="tool-icon">{'🔧'}</span>
          <span className="tool-name">{tool.tool}</span>
          {summary && <span className="tool-summary">{summary}</span>}
        </span>
        <span className={`tool-status ${tool.status || 'done'}`}>{tool.status || 'done'}</span>
      </button>
      {expanded && (
        <div className="tool-card-body">
          {tool.args && <pre>{JSON.stringify(tool.args, null, 2)}</pre>}
          {tool.result && <pre>{typeof tool.result === 'string' ? tool.result : JSON.stringify(tool.result, null, 2)}</pre>}
          {tool.error && <pre className="tool-error">{tool.error}</pre>}
        </div>
      )}
    </div>
  )
}

function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim() && !disabled) { onSend(text); setText('') }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e) }
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={disabled ? 'AI is thinking...' : 'Type a message... (Enter to send)'}
        rows={1}
        disabled={disabled}
      />
      <button type="submit" disabled={!text.trim() || disabled}>{disabled ? '...' : 'Send'}</button>
    </form>
  )
}

function fmtTime(iso) {
  if (!iso) return ''
  try {
    const d = new Date(iso)
    const now = new Date()
    const diff = now - d
    if (diff < 60000) return 'just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    return d.toLocaleDateString()
  } catch { return '' }
}
