import React, { useEffect, useState } from 'react'
import AssumptionsPanel from './components/AssumptionsPanel.jsx'
import Chat from './components/Chat.jsx'
import FlywheelPanel from './components/FlywheelPanel.jsx'
import { fetchConfig } from './api.js'

const initialTab = () =>
  (typeof location !== 'undefined' && location.hash.replace('#', '') === 'flywheel')
    ? 'flywheel' : 'chat'

export default function App() {
  const [tab, setTab] = useState(initialTab)
  const [flywheelEnabled, setFlywheelEnabled] = useState(false)
  const select = (id) => { setTab(id); if (typeof location !== 'undefined') location.hash = id }

  useEffect(() => {
    fetchConfig().then((c) => setFlywheelEnabled(!!c.flywheel_enabled)).catch(() => {})
  }, [])

  const tabs = [
    { id: 'chat', label: 'Campaign estimator' },
    { id: 'flywheel', label: flywheelEnabled ? 'Learning flywheel' : 'Learning flywheel (coming soon)' },
  ]

  return (
    <div className="app">
      <header className="topbar">
        <h1>It's Today Media — Lead Monetization</h1>
        <p className="sub">
          Turn leads you've already paid to acquire into revenue: the right offer to the
          right lead, and a model that gets smarter with every send.
        </p>
        <nav className="tabs">
          {tabs.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => select(t.id)}>{t.label}</button>
          ))}
        </nav>
      </header>

      {tab === 'chat' ? (
        <main className="layout">
          <AssumptionsPanel />
          <Chat />
        </main>
      ) : (
        <main className="layout-full">
          <FlywheelPanel enabled={flywheelEnabled} />
        </main>
      )}
    </div>
  )
}
