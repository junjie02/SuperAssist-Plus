const BASE = '/api'

let token = sessionStorage.getItem('access_token') || ''

export function setToken(t) {
  token = t
  sessionStorage.setItem('access_token', t)
}

export function getToken() {
  return token
}

export function clearToken() {
  token = ''
  sessionStorage.removeItem('access_token')
}

async function request(method, path, body) {
  const headers = { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })

  if (res.status === 401) {
    clearToken()
    window.location.href = '/login'
    return
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }

  return res.json()
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body),
  del: (path) => request('DELETE', path),
}

export async function login(username, password) {
  const data = await api.post('/auth/login', { username, password })
  setToken(data.access_token)
  return data
}

export async function register(username, password) {
  const data = await api.post('/auth/register', { username, password })
  setToken(data.access_token)
  return data
}
