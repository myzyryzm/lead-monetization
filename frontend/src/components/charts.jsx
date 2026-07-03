import React, { useRef, useState } from 'react'

// Categorical colors — follow the policy (entity), fixed order. Validated
// colorblind-safe against the dark surface via the dataviz skill's validator.
export const POLICY = {
  thompson: { color: '#3987e5', label: 'Thompson (bandit)' },
  greedy:   { color: '#199e70', label: 'Greedy (exploit-only)' },
  static:   { color: '#9085e9', label: 'Static (no flywheel)' },
  random:   { color: '#e66767', label: 'Random' },
  oracle:   { color: '#c3c2b7', label: 'Oracle (ceiling)', dashed: true },
}

const INK2 = '#9aa6b2'
const MUTED = '#6b7581'
const GRID = '#242b34'

// push overlapping right-edge labels apart, keep order
function spread(items, gap, lo, hi) {
  const a = items.map((d) => ({ ...d })).sort((x, y) => x.y - y.y)
  for (let i = 1; i < a.length; i++) {
    if (a[i].y - a[i - 1].y < gap) a[i].y = a[i - 1].y + gap
  }
  if (a.length && a[a.length - 1].y > hi) {
    const shift = a[a.length - 1].y - hi
    a.forEach((d) => { d.y = Math.max(lo, d.y - shift) })
  }
  return a
}

/**
 * Multi-series line chart with a hover crosshair + tooltip.
 * props: rounds[], series[{key,values}], yFmt, height, forceY0, upTo, yFmtTip
 */
export function LineChart({ rounds, series, yFmt, height = 320, forceY0 = true,
  upTo = null, yFmtTip = null }) {
  const W = 760
  const mL = 56, mR = 138, mT = 14, mB = 30
  const pw = W - mL - mR, ph = height - mT - mB
  const n = rounds.length
  const shown = upTo == null ? n : Math.max(1, Math.min(upTo, n))
  const svgRef = useRef(null)
  const [hi, setHi] = useState(null)

  const all = series.flatMap((s) => s.values)
  let ymin = forceY0 ? 0 : Math.min(...all)
  let ymax = Math.max(...all)
  if (ymax === ymin) ymax = ymin + 1
  ymax += (ymax - ymin) * 0.06
  const px = (i) => mL + (n > 1 ? (pw * i) / (n - 1) : 0)
  const py = (v) => mT + ph * (1 - (v - ymin) / (ymax - ymin))
  const fmtTip = yFmtTip || yFmt

  const yticks = 5
  const grid = []
  for (let k = 0; k <= yticks; k++) {
    const v = ymin + ((ymax - ymin) * k) / yticks
    const y = py(v)
    grid.push(
      <g key={k}>
        <line x1={mL} y1={y} x2={mL + pw} y2={y} stroke={GRID} strokeWidth="1" />
        <text x={mL - 8} y={y + 3.5} textAnchor="end" fill={MUTED} fontSize="11"
          style={{ fontVariantNumeric: 'tabular-nums' }}>{yFmt(v)}</text>
      </g>
    )
  }
  const xstep = Math.max(1, Math.floor((n - 1) / 6))
  const xt = []
  for (let i = 0; i < n; i += xstep) {
    xt.push(<text key={i} x={px(i)} y={mT + ph + 20} textAnchor="middle"
      fill={MUTED} fontSize="11">{rounds[i]}</text>)
  }

  // labels at right edge (stable legend), based on final values
  const endLabels = spread(
    series.map((s) => ({ key: s.key, y: py(s.values[n - 1]) })),
    15, mT + 6, mT + ph
  )

  function onMove(e) {
    const r = svgRef.current.getBoundingClientRect()
    const vbx = ((e.clientX - r.left) / r.width) * W
    let idx = Math.round(((vbx - mL) / pw) * (n - 1))
    idx = Math.max(0, Math.min(shown - 1, idx))
    if (vbx < mL - 4 || vbx > mL + pw + 4) { setHi(null); return }
    setHi(idx)
  }

  return (
    <div className="chartbox">
      <svg ref={svgRef} viewBox={`0 0 ${W} ${height}`} width="100%"
        preserveAspectRatio="xMidYMid meet" role="img"
        onMouseMove={onMove} onMouseLeave={() => setHi(null)}
        style={{ fontFamily: 'system-ui,-apple-system,Segoe UI,sans-serif' }}>
        {grid}{xt}
        <text x={mL + pw / 2} y={height - 2} textAnchor="middle" fill={MUTED}
          fontSize="10.5">round</text>

        {series.map((s) => {
          const p = POLICY[s.key]
          const pts = s.values.slice(0, shown)
            .map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')
          return (
            <polyline key={s.key} points={pts} fill="none" stroke={p.color}
              strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
              strokeDasharray={p.dashed ? '5 4' : undefined} />
          )
        })}

        {/* moving end dots */}
        {series.map((s) => {
          const p = POLICY[s.key]
          return <circle key={s.key} cx={px(shown - 1)} cy={py(s.values[shown - 1])}
            r="3.2" fill={p.color} />
        })}

        {/* right-edge direct labels (colored dot + ink text) */}
        {endLabels.map((d) => (
          <g key={d.key}>
            <circle cx={mL + pw + 12} cy={d.y - 3} r="3.4" fill={POLICY[d.key].color} />
            <text x={mL + pw + 20} y={d.y} fill={INK2} fontSize="11.5">
              {POLICY[d.key].label}</text>
          </g>
        ))}

        {/* hover crosshair */}
        {hi != null && (
          <g pointerEvents="none">
            <line x1={px(hi)} y1={mT} x2={px(hi)} y2={mT + ph} stroke={MUTED}
              strokeWidth="1" strokeDasharray="3 3" />
            {series.map((s) => (
              <circle key={s.key} cx={px(hi)} cy={py(s.values[hi])} r="3.6"
                fill={POLICY[s.key].color} stroke="#0f1216" strokeWidth="1.5" />
            ))}
          </g>
        )}
      </svg>

      {hi != null && (
        <div className="chart-tip" style={{
          left: `${((px(hi) + (px(hi) > mL + pw / 2 ? -8 : 8)) / W) * 100}%`,
          transform: px(hi) > mL + pw / 2 ? 'translateX(-100%)' : 'none',
        }}>
          <div className="tt-h">round {rounds[hi]}</div>
          {series.map((s) => (
            <div className="tt-r" key={s.key}>
              <span className="tt-dot" style={{ background: POLICY[s.key].color }} />
              <span className="tt-l">{POLICY[s.key].label}</span>
              <span className="tt-v">{fmtTip(s.values[hi])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function mix(a, b, t) {
  t = Math.max(0, Math.min(1, t))
  const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))
  const [ar, ag, ab] = p(a), [br, bg, bb] = p(b)
  const c = (x, y) => Math.round(x + (y - x) * t).toString(16).padStart(2, '0')
  return `#${c(ar, br)}${c(ag, bg)}${c(ab, bb)}`
}

/** 5x5 EV-error heatmap. props: matrix, rows, cols, vmax */
export function Heatmap({ matrix, rows, cols, vmax }) {
  const cell = 46, lead = 76, top = 22
  const W = lead + cols.length * cell + 6
  const H = top + rows.length * cell + 8
  const LO = '#1a212b', HIu = '#5aa0f0'
  const [hv, setHv] = useState(null)
  return (
    <div className="chartbox heatbox">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}
        style={{ maxWidth: '100%', fontFamily: 'system-ui,sans-serif' }}>
        {cols.map((c, ci) => (
          <text key={c} x={lead + ci * cell + cell / 2} y={top - 7} textAnchor="middle"
            fill={MUTED} fontSize="9.5">{c.slice(0, 4)}</text>
        ))}
        {rows.map((r, ri) => (
          <g key={r}>
            <text x={lead - 8} y={top + ri * cell + cell / 2 + 3} textAnchor="end"
              fill={INK2} fontSize="10.5">{r.replace(/_/g, ' ').slice(0, 9)}</text>
            {cols.map((c, ci) => {
              const v = matrix[ri][ci]
              const t = vmax > 0 ? v / vmax : 0
              const x = lead + ci * cell, y = top + ri * cell
              const active = hv && hv.r === ri && hv.c === ci
              return (
                <g key={c} onMouseEnter={() => setHv({ r: ri, c: ci, v })}
                  onMouseLeave={() => setHv(null)}>
                  <rect x={x + 1} y={y + 1} width={cell - 2} height={cell - 2} rx="3"
                    fill={mix(LO, HIu, t)} stroke={active ? '#e7ecf2' : 'none'}
                    strokeWidth="1.5" />
                  <text x={x + cell / 2} y={y + cell / 2 + 3.5} textAnchor="middle"
                    fill={t > 0.55 ? '#e7ecf2' : INK2} fontSize="10"
                    style={{ fontVariantNumeric: 'tabular-nums' }}>{v.toFixed(1)}</text>
                </g>
              )
            })}
          </g>
        ))}
      </svg>
      {hv && (
        <div className="heat-cap">
          {rows[hv.r].replace(/_/g, ' ')} intent × {cols[hv.c]} offer —
          &nbsp;EV-error <b>${hv.v.toFixed(2)}</b>
        </div>
      )}
    </div>
  )
}
