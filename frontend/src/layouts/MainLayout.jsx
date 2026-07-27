import { NavLink, useLocation, useNavigate } from 'react-router-dom'
import { Bot, Library, LogOut, MessageSquare, Settings, Share2 } from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import ChatPage from '../pages/ChatPage'
import GraphPage from '../pages/GraphPage'
import SettingsPage from '../pages/SettingsPage'
import KnowledgePage from '../pages/KnowledgePage'

export default function MainLayout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const showGraph = location.pathname.startsWith('/graph')
  const showSettings = location.pathname.startsWith('/settings')
  const showKnowledge = location.pathname.startsWith('/knowledge')
  const showChat = !showGraph && !showSettings && !showKnowledge

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="app-shell">
      <nav className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo"><Bot size={22} aria-hidden="true" /></div>
          <div className="sidebar-user">{user?.username || 'User'}</div>
        </div>

        <ul className="sidebar-nav">
          <li>
            <NavLink to="/" end className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
              <MessageSquare size={18} aria-hidden="true" />
              <span>Chat</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/graph" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
              <Share2 size={18} aria-hidden="true" />
              <span>Memory Graph</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/knowledge" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
              <Library size={18} aria-hidden="true" />
              <span>Knowledge</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/settings" className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
              <Settings size={18} aria-hidden="true" />
              <span>Settings</span>
            </NavLink>
          </li>
        </ul>

        <div className="sidebar-footer">
          <button onClick={handleLogout} className="logout-btn">
            <LogOut size={17} aria-hidden="true" />
            <span>Logout</span>
          </button>
        </div>
      </nav>

      <main className="main-content">
        <div style={{ display: showChat ? undefined : 'none' }}>
          <ChatPage />
        </div>
        <div style={{ display: showGraph ? undefined : 'none' }}>
          <GraphPage />
        </div>
        {showKnowledge && <KnowledgePage />}
        {showSettings && <SettingsPage />}
      </main>
    </div>
  )
}
