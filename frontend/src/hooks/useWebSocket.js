import { useRef, useCallback, useEffect, useState } from 'react'
import { getToken } from '../lib/api'

export default function useWebSocket() {
  const wsRef = useRef(null)
  const handlersRef = useRef({})
  const [ready, setReady] = useState(false)

  const connect = useCallback(() => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setReady(false)
    const token = getToken()
    if (!token) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat?token=${token}`)

    ws.onopen = () => {
      setReady(true)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const handler = handlersRef.current[data.type]
        if (handler) handler(data)
        // Fallback: call 'all' handler
        const all = handlersRef.current['*']
        if (all && data.type !== 'agent_text') all(data)
      } catch { /* ignore malformed messages */ }
    }

    ws.onclose = () => {
      setReady(false)
      wsRef.current = null
      // Auto-reconnect after 2 seconds
      setTimeout(() => {
        if (!wsRef.current) connect()
      }, 2000)
    }

    ws.onerror = () => {
      ws?.close()
    }

    wsRef.current = ws
  }, [])

  const send = useCallback((message, threadId = null, ragMode = false) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false
    }
    wsRef.current.send(JSON.stringify({ message, thread_id: threadId, rag_mode: ragMode }))
    return true
  }, [])

  const on = useCallback((type, handler) => {
    handlersRef.current[type] = handler
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { send, on, connect, ready }
}
