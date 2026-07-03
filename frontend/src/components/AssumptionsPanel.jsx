import React, { useEffect, useState } from 'react'
import { fetchAssumptions } from '../api.js'

export default function AssumptionsPanel() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetchAssumptions().then(setData).catch((e) => setErr(e.message))
  }, [])

  if (err) return <aside className="panel">Couldn't load assumptions: {err}</aside>
  if (!data) return <aside className="panel">Loading assumptions…</aside>

  return (
    <aside className="panel">
      <h2>How this works</h2>
      <p className="muted">
        {data.n_leads.toLocaleString()} leads · {data.n_offers} offers ·{' '}
        {data.categories.length} categories
      </p>

      <ul className="notes">
        {data.notes.map((n, i) => <li key={i}>{n}</li>)}
      </ul>

      <h3>Offer categories</h3>
      <div className="cats">
        {data.categories.map((c) => (
          <div className="cat" key={c.category}>
            <div className="cat-head">
              <span className="cat-name">{c.category.replace(/_/g, ' ')}</span>
              <span className="cat-payout">${c.payout_min}–${c.payout_max}</span>
            </div>
            <ul className="offers">
              {c.offers.map((o) => (
                <li key={o.offer_id}>
                  <span>{o.offer_name}</span>
                  <span className="offer-meta">{o.commitment_label} · ${o.payout}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <h3>Commitment ladder</h3>
      <ul className="ladder">
        {Object.entries(data.commitment_ladder).map(([lvl, label]) => (
          <li key={lvl}><b>{lvl}</b> · {label}</li>
        ))}
      </ul>
    </aside>
  )
}
