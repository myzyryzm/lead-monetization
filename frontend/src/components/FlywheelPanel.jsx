import React, { useEffect, useRef, useState } from 'react'
import { runFlywheel } from '../api.js'
import { LineChart, Heatmap, POLICY } from './charts.jsx'

const ORDER = ['oracle', 'thompson', 'greedy', 'static', 'random']

function money(v) {
  const a = Math.abs(v)
  if (a >= 10000) return `$${(v / 1000).toFixed(0)}k`
  if (a >= 1000) return `$${(v / 1000).toFixed(1)}k`
  return `$${v.toFixed(0)}`
}
const moneyFull = (v) => `$${Math.round(v).toLocaleString()}`
const pct = (v) => `${v.toFixed(0)}%`

function Field({ label, value, set, min, max, step = 1 }) {
  return (
    <label className="fw-field">
      <span>{label}</span>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => set(e.target.value)} />
    </label>
  )
}

const POLICY_BLURBS = [
  ['oracle', 'sends by the true expected value — the perfect-information ceiling'],
  ['thompson', 'the flywheel plus a Thompson-sampling bandit that explores where it is uncertain'],
  ['greedy', 'the flywheel, exploit-only — retrain, then send the model’s current top picks'],
  ['static', 'trained once and frozen — isolates “no flywheel”'],
  ['random', 'random sends — the floor'],
]

export default function FlywheelPanel({ enabled = false }) {
  const [cfg, setCfg] = useState({ rounds: 24, budget: 400, ensemble: 8, seed: 7 })
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [play, setPlay] = useState(0)          // rounds revealed so far
  const timer = useRef(null)

  function animate(n) {
    clearInterval(timer.current)
    setPlay(1)
    const stepMs = Math.max(35, Math.round(2400 / n))
    timer.current = setInterval(() => {
      setPlay((p) => {
        if (p >= n) { clearInterval(timer.current); return n }
        return p + 1
      })
    }, stepMs)
  }

  async function run() {
    if (busy) return
    setBusy(true); setErr(null)
    clearInterval(timer.current)
    try {
      const data = await runFlywheel({
        rounds: +cfg.rounds, budget: +cfg.budget,
        ensemble: +cfg.ensemble, seed: +cfg.seed,
      })
      setRes(data)
      animate(data.rounds.length)
    } catch (e) {
      setErr(e.message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => { if (enabled) run() }, [enabled])   // auto-run once enabled
  useEffect(() => () => clearInterval(timer.current), [])

  if (!enabled) {
    return (
      <section className="fw">
        <div className="fw-intro">
          <h2>The learning flywheel <span className="soon-badge">Coming soon</span></h2>
          <p>
            A static model trains once and stops. In production you only learn the outcome of a
            (lead, offer) you actually <em>send</em> — so every send is training data and the model
            should compound: send → observe conversions → retrain → target better next time. The catch
            is that the allocation policy shapes its own training data. A pure-greedy policy stops
            probing segments it currently thinks are weak, so it never collects the data that would
            correct a wrong belief — it can trap itself. The fix is <strong>exploration</strong>.
          </p>
        </div>

        <div className="fw-card">
          <h3>What it will do</h3>
          <p className="sub">
            Race five allocation policies over many rounds against the engine’s own hidden-truth world,
            and watch which one earns the most:
          </p>
          <ul className="fw-policylist">
            {POLICY_BLURBS.map(([k, blurb]) => (
              <li key={k}>
                <span className="pdot" style={{ background: POLICY[k].color }} />
                <b>{POLICY[k].label}</b> — {blurb}
              </li>
            ))}
          </ul>
        </div>

        <div className="fw-card">
          <h3>What it demonstrates</h3>
          <ul className="fw-bullets">
            <li>The flywheel compounds — retraining on your own sends captures ~90% of a perfect
              oracle’s revenue, beating a frozen model by ~40% and random blasting by ~4–5×.</li>
            <li>Exploration is insurance — a Thompson-sampling bandit keeps the model accurate in
              every segment, with no blind spots, so it holds up when the offer mix shifts.</li>
            <li>Live charts: a cumulative-revenue race, share of the oracle ceiling, model-error over
              time, and a per-segment “blind-spot” heatmap.</li>
          </ul>
        </div>

        <p className="fw-foot">
          This tab is turned off for now. To enable the interactive simulation, run the server with
          <code> ENABLE_FLYWHEEL=true</code>. The engine already exists as a standalone
          <code> flywheel.py</code>, which also writes a shareable HTML report.
        </p>
      </section>
    )
  }

  const sm = res?.summary
  const done = res && play >= res.rounds.length
  const kpis = sm && [
    ['Flywheel vs. frozen model',
      `+${(100 * (sm.greedy.total_revenue / Math.max(sm.static.total_revenue, 1) - 1)).toFixed(0)}%`,
      'greedy retrains on its own sends; static never does'],
    ['Best learner vs. oracle', `${sm.thompson.pct_of_oracle.toFixed(0)}%`,
      'of the perfect-information maximum, captured'],
    ['Model error, explorer vs. greedy',
      `−${(100 * (1 - (sm.thompson.mean_ev_error || 0) / Math.max(sm.greedy.mean_ev_error || 1, 1e-9))).toFixed(0)}%`,
      'a more accurate model in every segment'],
    ['Revenue vs. random blasting',
      `${(sm.thompson.total_revenue / Math.max(sm.random.total_revenue, 1)).toFixed(1)}×`,
      'same send budget, targeted by the learned model'],
  ]

  return (
    <section className="fw">
      <div className="fw-intro">
        <h2>The learning flywheel</h2>
        <p>
          A static model trains once and stops. In production you only learn the outcome of a
          (lead, offer) you actually <em>send</em> — so every send is training data and the model
          should compound. But the allocation policy shapes its own data: a pure-greedy policy stops
          probing segments it thinks are weak and can trap itself. This runs that loop against the
          engine’s hidden-truth world and races five policies.
        </p>
      </div>

      <div className="fw-controls">
        <Field label="rounds" value={cfg.rounds} min={4} max={36}
          set={(v) => setCfg({ ...cfg, rounds: v })} />
        <Field label="sends / round" value={cfg.budget} min={50} max={800} step={50}
          set={(v) => setCfg({ ...cfg, budget: v })} />
        <Field label="bandit heads" value={cfg.ensemble} min={2} max={12}
          set={(v) => setCfg({ ...cfg, ensemble: v })} />
        <Field label="seed" value={cfg.seed} min={0} max={9999}
          set={(v) => setCfg({ ...cfg, seed: v })} />
        <button className="fw-run" onClick={run} disabled={busy}>
          {busy ? 'Running…' : 'Run simulation'}
        </button>
        {done && (
          <button className="fw-replay" onClick={() => animate(res.rounds.length)}>Replay</button>
        )}
      </div>

      {err && <div className="fw-err">Error: {err}</div>}

      {busy && !res && (
        <div className="fw-loading">Running {cfg.rounds} rounds of the learning loop… (~10s)</div>
      )}

      {res && (
        <>
          <div className="fw-kpis">
            {kpis.map(([t, v, s]) => (
              <div className="fw-kpi" key={t}>
                <div className="kt">{t}</div><div className="kv">{v}</div><div className="ks">{s}</div>
              </div>
            ))}
          </div>

          <div className="fw-card">
            <h3>Cumulative revenue, round by round</h3>
            <p className="sub">Both learners pull away from the frozen model and leave random
              blasting far below. Oracle (dashed) is the perfect-information ceiling. Hover for values.</p>
            <LineChart rounds={res.rounds} yFmt={money} yFmtTip={moneyFull} upTo={play}
              series={ORDER.map((k) => ({ key: k, values: res.series[k].cum_revenue }))} />
          </div>

          <div className="fw-card">
            <h3>Share of the oracle ceiling captured</h3>
            <p className="sub">The flywheel policies climb toward ~90% of what a perfect oracle would
              earn; the frozen and random policies plateau well below.</p>
            <LineChart rounds={res.rounds} yFmt={pct} yFmtTip={(v) => `${v.toFixed(1)}%`} upTo={play}
              series={ORDER.filter((k) => k !== 'oracle')
                .map((k) => ({ key: k, values: res.series[k].pct_of_oracle }))} />
          </div>

          <div className="fw-card">
            <h3>How wrong is each model? <span className="dim">mean EV-error over all lead×offer pairs</span></h3>
            <p className="sub">Lower is better. The explorer’s model is the most accurate, the frozen
              one the least — the payoff of exploration is a model with no blind spots.</p>
            <LineChart rounds={res.rounds} yFmt={(v) => `$${v.toFixed(1)}`}
              yFmtTip={(v) => `$${v.toFixed(2)}`} upTo={play} forceY0
              series={['static', 'greedy', 'thompson']
                .map((k) => ({ key: k, values: res.series[k].ev_error }))} />
          </div>

          {done && (
            <div className="fw-card">
              <h3>Where each model is wrong — the blind-spot map</h3>
              <p className="sub">Mean EV-error per lead-intent (row) × offer-category (column), final
                round. Brighter = the dollar estimate is further off. Greedy develops hot cells in
                segments it under-samples; the explorer stays uniformly cool. Hover a cell.</p>
              <div className="fw-heatrow">
                <figure><figcaption>Greedy (exploit-only)</figcaption>
                  <Heatmap matrix={res.heatmaps.greedy} rows={res.heatmaps.intents}
                    cols={res.heatmaps.offer_categories} vmax={res.heatmaps.max} /></figure>
                <figure><figcaption>Thompson (bandit)</figcaption>
                  <Heatmap matrix={res.heatmaps.thompson} rows={res.heatmaps.intents}
                    cols={res.heatmaps.offer_categories} vmax={res.heatmaps.max} /></figure>
                <div className="fw-cbar">
                  <span>${res.heatmaps.max.toFixed(1)}</span>
                  <div className="bar" /><span>$0</span><small>EV-error</small>
                </div>
              </div>
            </div>
          )}

          {done && (
            <div className="fw-card">
              <h3>The claims, checked against the numbers</h3>
              <ul className="fw-checks">
                {res.assertions.map((a, i) => (
                  <li key={i} className={a.pass ? 'ok' : 'no'}>
                    <span className="badge">{a.pass ? 'PASS' : '—'}</span>
                    <span className="cn">{a.name}</span>
                    <span className="cd">{a.detail}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="fw-foot">
            All data is synthetic — this demonstrates the learning architecture, not results on real
            campaign data. The standalone <code>flywheel.py</code> writes the same charts to a shareable
            HTML report.
          </p>
        </>
      )}
    </section>
  )
}
