import React from 'react'
import AssumptionsPanel from './components/AssumptionsPanel.jsx'
import Chat from './components/Chat.jsx'

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <h1>It's Today Media — Lead Monetization</h1>
        <p className="sub">
          Turn leads you've already paid to acquire into revenue: the right offer to the
          right lead.
        </p>
      </header>

      <main className="layout">
        <AssumptionsPanel />
        <Chat />
      </main>
    </div>
  )
}
