import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { api, clearToken, getToken, setToken } from '../lib/api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!getToken()) {
      setLoading(false)
      return
    }
    api.get('/auth/me')
      .then(data => setUser({ id: data.user_id, username: data.username, isAdmin: data.is_admin === true }))
      .catch(() => clearToken())
      .finally(() => setLoading(false))
  }, [])

  const logout = useCallback(() => {
    clearToken()
    setUser(null)
  }, [])

  const loginUser = useCallback((data) => {
    setToken(data.access_token)
    setUser({ id: data.user_id, username: data.username, isAdmin: data.is_admin === true })
  }, [])

  return (
    <AuthContext.Provider value={{ user, loading, loginUser, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
