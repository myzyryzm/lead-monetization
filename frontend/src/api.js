export async function fetchAssumptions() {
  const r = await fetch('/api/assumptions')
  if (!r.ok) throw new Error(`assumptions failed (HTTP ${r.status})`)
  return r.json()
}

export async function fetchConfig() {
  const r = await fetch('/api/config')
  if (!r.ok) throw new Error(`config failed (HTTP ${r.status})`)
  return r.json()
}

export async function sendChat(messages) {
  const r = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`)
  return data
}

export async function runFlywheel(config) {
  const r = await fetch('/api/flywheel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`)
  return data
}
