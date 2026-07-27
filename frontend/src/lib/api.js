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
  const isFormData = body instanceof FormData
  const headers = isFormData ? {} : { 'Content-Type': 'application/json' }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? (isFormData ? body : JSON.stringify(body)) : undefined,
  })

  if (res.status === 401) {
    clearToken()
    window.location.href = '/login'
    return
  }

  const contentType = res.headers.get('content-type') || ''
  const responseText = await res.text()
  let data = {}
  if (responseText) {
    if (!contentType.includes('application/json')) {
      throw new Error(`Unexpected non-JSON response from server (HTTP ${res.status}).`)
    }
    try {
      data = JSON.parse(responseText)
    } catch {
      throw new Error(`Invalid JSON response from server (HTTP ${res.status}).`)
    }
  }

  if (!res.ok) {
    const err = data
    const detail = Array.isArray(err.detail)
      ? err.detail.map(item => item.msg || item.error || String(item)).join('; ')
      : err.detail
    throw new Error(detail || `HTTP ${res.status}`)
  }

  return data
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body),
  put: (path, body) => request('PUT', path, body),
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
