import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import ChatPage from '../pages/ChatPage'
import GraphPage from '../pages/GraphPage'

export default function MainLayout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const showGraph = location.pathname.startsWith('/graph')

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">🧠</div>
          <div className="sidebar-user">{user?.username || 'User'}</div>
        </div>

        <ul className="sidebar-nav">
          <li>
            <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
              💬 Chat
            </NavLink>
          </li>
          <li>
            <NavLink to="/graph" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
              🧠 Memory Graph
            </NavLink>
          </li>
        </ul>

        <div className="sidebar-footer">
          <button onClick={handleLogout} className="logout-btn">Logout</button>
        </div>
      </nav>

      <main className="main-content">
        <div style={{ display: showGraph ? 'none' : undefined }}>
          <ChatPage />
        </div>
        <div style={{ display: showGraph ? undefined : 'none' }}>
          <GraphPage />
        </div>
      </main>
    </div>
  )
}
