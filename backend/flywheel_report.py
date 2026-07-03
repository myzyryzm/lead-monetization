"""
flywheel_report.py — render a flywheel `simulate()` result as a self-contained,
dependency-free HTML report (inline SVG charts, no JS libraries, no external
assets). Committed to a dark look to match the app's aesthetic.

Charts (colors validated colorblind-safe via the dataviz skill's validator):
  1. Cumulative revenue race        — all 5 policies
  2. % of the oracle ceiling captured — all 5 policies
  3. Model EV-error over rounds      — the 3 policies that hold a model
  4. Blind-spot heatmaps             — per-segment EV-error, greedy vs thompson

Entry point: render(results_dict) -> html string.
"""

# --- theme (dark, matches the SPA) -----------------------------------------
PAGE = "#0f1216"
PANEL = "#171b21"
PANEL2 = "#1e242c"
GRID = "#242b34"
AXIS = "#39424e"
INK = "#e7ecf2"
INK2 = "#9aa6b2"
MUTED = "#6b7581"
OK = "#2fbf87"

# categorical: color follows the policy (entity), fixed order — never rank
COLORS = {
    "thompson": "#3987e5",   # blue  — the recommended policy (hero)
    "greedy":   "#199e70",   # aqua
    "static":   "#9085e9",   # violet
    "random":   "#e66767",   # red   — the floor
    "oracle":   "#c3c2b7",   # neutral, dashed — the theoretical ceiling
}
DASHED = {"oracle"}
LABEL = {"thompson": "Thompson (bandit)", "greedy": "Greedy (exploit-only)",
         "static": "Static (no flywheel)", "random": "Random", "oracle": "Oracle (ceiling)"}


# --- small helpers ----------------------------------------------------------
def _money(v):
    v = float(v)
    if abs(v) >= 1000:
        return f"${v/1000:.0f}k" if abs(v) >= 10000 else f"${v/1000:.1f}k"
    return f"${v:.0f}"


def _hexmix(a, b, t):
    t = max(0.0, min(1.0, t))
    ai = [int(a[i:i+2], 16) for i in (1, 3, 5)]
    bi = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(ai[k] + (bi[k]-ai[k])*t):02x}" for k in range(3))


def _spread(labels, min_gap=15.0, lo=0.0, hi=1e9):
    """Push overlapping right-edge labels apart while keeping their order."""
    labels = sorted(labels, key=lambda d: d["y"])
    for i in range(1, len(labels)):
        if labels[i]["y"] - labels[i-1]["y"] < min_gap:
            labels[i]["y"] = labels[i-1]["y"] + min_gap
    # clamp block if it overflowed the bottom
    if labels and labels[-1]["y"] > hi:
        shift = labels[-1]["y"] - hi
        for d in labels:
            d["y"] = max(lo, d["y"] - shift)
    return labels


# --- line chart -------------------------------------------------------------
def _line_chart(rounds, lines, y_fmt, *, force_y0=True, width=760, height=340,
                y_ticks=5):
    """lines: list of {"key", "values"}. Returns an <svg> string."""
    mL, mR, mT, mB = 62, 148, 16, 34
    pw, ph = width - mL - mR, height - mT - mB

    allv = [v for ln in lines for v in ln["values"]]
    ymin = 0.0 if force_y0 else min(allv)
    ymax = max(allv)
    if ymax == ymin:
        ymax = ymin + 1
    ymax += (ymax - ymin) * 0.06

    n = len(rounds)
    def px(i): return mL + (pw * (i / (n - 1) if n > 1 else 0))
    def py(v): return mT + ph * (1 - (v - ymin) / (ymax - ymin))

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
           f'preserveAspectRatio="xMidYMid meet" role="img" '
           f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">']

    # horizontal gridlines + y ticks
    for k in range(y_ticks + 1):
        v = ymin + (ymax - ymin) * k / y_ticks
        y = py(v)
        out.append(f'<line x1="{mL}" y1="{y:.1f}" x2="{mL+pw}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{mL-8}" y="{y+3.5:.1f}" text-anchor="end" '
                   f'fill="{MUTED}" font-size="11" '
                   f'style="font-variant-numeric:tabular-nums">{y_fmt(v)}</text>')

    # x ticks (~6)
    step = max(1, (n - 1) // 6)
    for i in range(0, n, step):
        x = px(i)
        out.append(f'<text x="{x:.1f}" y="{mT+ph+20:.1f}" text-anchor="middle" '
                   f'fill="{MUTED}" font-size="11">{rounds[i]}</text>')
    out.append(f'<text x="{mL+pw/2:.1f}" y="{height-1:.0f}" text-anchor="middle" '
               f'fill="{MUTED}" font-size="10.5">round</text>')

    # series
    end_labels = []
    for ln in lines:
        key = ln["key"]
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ln["values"]))
        dash = ' stroke-dasharray="5 4"' if key in DASHED else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[key]}" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
        end_labels.append({"key": key, "y": py(ln["values"][-1]),
                           "yr": py(ln["values"][-1])})
    # end dots at true positions
    for d in end_labels:
        out.append(f'<circle cx="{mL+pw:.1f}" cy="{d["yr"]:.1f}" r="3.2" '
                   f'fill="{COLORS[d["key"]]}"/>')
    # decluttered direct labels (colored dot + ink text)
    for d in _spread(end_labels, 15.0, mT + 6, mT + ph):
        ly = d["y"]
        out.append(f'<circle cx="{mL+pw+12:.1f}" cy="{ly-3:.1f}" r="3.4" '
                   f'fill="{COLORS[d["key"]]}"/>')
        out.append(f'<text x="{mL+pw+20:.1f}" y="{ly:.1f}" fill="{INK2}" '
                   f'font-size="11.5">{LABEL[d["key"]]}</text>')
    out.append("</svg>")
    return "".join(out)


# --- heatmap ----------------------------------------------------------------
def _heatmap(matrix, rows, cols, vmax, *, cell=44):
    lead = 74           # left gutter for row labels
    top = 22            # top gutter for col labels
    w = lead + len(cols) * cell + 6
    h = top + len(rows) * cell + 8
    lo, hi = "#1a212b", "#5aa0f0"     # low error -> near-surface, high -> bright
    out = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="system-ui,-apple-system,sans-serif">']
    for c, cl in enumerate(cols):
        cx = lead + c * cell + cell / 2
        out.append(f'<text x="{cx:.1f}" y="{top-7}" text-anchor="middle" '
                   f'fill="{MUTED}" font-size="9.5">{cl[:4]}</text>')
    for r, rl in enumerate(rows):
        ry = top + r * cell + cell / 2
        out.append(f'<text x="{lead-8}" y="{ry+3:.1f}" text-anchor="end" '
                   f'fill="{INK2}" font-size="10.5">{rl.replace("_"," ")[:9]}</text>')
        for c in range(len(cols)):
            v = matrix[r][c]
            t = (v / vmax) if vmax > 0 else 0
            fill = _hexmix(lo, hi, t)
            x = lead + c * cell
            y = top + r * cell
            tcol = INK if t > 0.55 else INK2
            out.append(f'<rect x="{x+1}" y="{y+1}" width="{cell-2}" height="{cell-2}" '
                       f'rx="3" fill="{fill}"><title>{rows[r]} × {cols[c]}: '
                       f'EV-error ${v:.2f}</title></rect>')
            out.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+3.5:.1f}" '
                       f'text-anchor="middle" fill="{tcol}" font-size="10" '
                       f'style="font-variant-numeric:tabular-nums">{v:.1f}</text>')
    out.append("</svg>")
    return "".join(out)


def _chart_card(title, sub, svg, note=""):
    note_html = f'<p class="cnote">{note}</p>' if note else ""
    return (f'<section class="card"><h3>{title}</h3>'
            f'<p class="csub">{sub}</p><div class="plot">{svg}</div>{note_html}</section>')


# --- the report -------------------------------------------------------------
def render(res):
    cfg = res["config"]
    sm = res["summary"]
    series = res["series"]
    rounds = res["rounds"]

    fly_lift = 100 * (sm["greedy"]["total_revenue"] / max(sm["static"]["total_revenue"], 1) - 1)
    vs_random = sm["thompson"]["total_revenue"] / max(sm["random"]["total_revenue"], 1)
    err_cut = 100 * (1 - (sm["thompson"]["mean_ev_error"] or 0) /
                     max(sm["greedy"]["mean_ev_error"] or 1, 1e-9))
    err_cut_static = 100 * (1 - (sm["thompson"]["mean_ev_error"] or 0) /
                            max(sm["static"]["mean_ev_error"] or 1, 1e-9))

    kpis = [
        ("Flywheel vs. frozen model", f"+{fly_lift:.0f}%",
         "greedy retrains on its own sends; static never does"),
        ("Best learner vs. oracle", f"{sm['thompson']['pct_of_oracle']:.0f}%",
         "of the theoretical-maximum revenue, captured"),
        ("Model error, explorer vs. greedy", f"−{err_cut:.0f}%",
         f"and −{err_cut_static:.0f}% vs. the static model"),
        ("Revenue vs. random blasting", f"{vs_random:.1f}×",
         "same send budget, targeted by the learned model"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kt">{t}</div><div class="kv">{v}</div>'
        f'<div class="ks">{s}</div></div>' for t, v, s in kpis)

    # chart 1: cumulative revenue
    c1 = _line_chart(rounds, [{"key": p, "values": series[p]["cum_revenue"]}
                              for p in res["policies"]], _money)
    # chart 2: % of oracle (drop oracle's flat 100 line for clarity)
    c2 = _line_chart(rounds, [{"key": p, "values": series[p]["pct_of_oracle"]}
                              for p in res["policies"] if p != "oracle"],
                     lambda v: f"{v:.0f}%")
    # chart 3: EV-error over rounds (model-holding policies only)
    c3 = _line_chart(rounds, [{"key": p, "values": series[p]["ev_error"]}
                              for p in ("static", "greedy", "thompson")],
                     lambda v: f"${v:.1f}", force_y0=True)

    hm = res["heatmaps"]
    vmax = hm["max"]
    heat_g = _heatmap(hm["greedy"], hm["intents"], hm["offer_categories"], vmax)
    heat_t = _heatmap(hm["thompson"], hm["intents"], hm["offer_categories"], vmax)
    heat_html = (
        '<section class="card"><h3>Where each model is wrong — the blind-spot map</h3>'
        '<p class="csub">Mean EV-error per lead-intent (row) × offer-category (column) cell, '
        'final round. Brighter = the model’s dollar estimate is further off. '
        'The greedy model develops hot cells in segments it under-samples; the explorer stays '
        'uniformly cool.</p>'
        f'<div class="heatrow">'
        f'<figure><figcaption>Greedy (exploit-only)</figcaption>{heat_g}</figure>'
        f'<figure><figcaption>Thompson (bandit)</figcaption>{heat_t}</figure>'
        f'<div class="cbar"><span>${vmax:.1f}</span>'
        f'<div class="bar"></div><span>$0</span><small>EV-error</small></div>'
        '</div></section>')

    # thesis checks
    checks = "".join(
        f'<li class="{"ok" if a["pass"] else "no"}">'
        f'<span class="badge">{"PASS" if a["pass"] else "—"}</span>'
        f'<span class="cn">{a["name"]}</span>'
        f'<span class="cd">{a["detail"]}</span></li>'
        for a in res["assertions"])

    # summary table (accessibility / full data)
    head = "".join(f"<th>{h}</th>" for h in
                   ["Policy", "Revenue", "Conversions", "% of oracle",
                    "Coverage", "Mean EV-error"])
    body = ""
    for p in res["policies"]:
        s = sm[p]
        err = "—" if s["mean_ev_error"] is None else f"${s['mean_ev_error']:.2f}"
        body += (f'<tr><td><span class="dot" style="background:{COLORS[p]}"></span>'
                 f'{LABEL[p]}</td><td>${s["total_revenue"]:,.0f}</td>'
                 f'<td>{s["total_conversions"]:,}</td><td>{s["pct_of_oracle"]:.1f}%</td>'
                 f'<td>{s["final_coverage"]}/25</td><td>{err}</td></tr>')

    cfg_chips = " · ".join([
        f"{cfg['rounds']} rounds", f"{cfg['budget']} sends/round",
        f"ensemble {cfg['ensemble']}", f"warm-start {cfg['warmstart']}",
        f"retrain every {cfg['retrain_every']}",
        f"postback {cfg['postback']:.2f}", f"seed {cfg['seed']}",
        f"{cfg['reps']} rep" + ("s" if cfg["reps"] != 1 else "")])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lead-Monetization Flywheel — simulation report</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:{PAGE};color:{INK};
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 64px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}}
.lede{{color:{INK2};max-width:78ch;margin:0 0 4px}}
.chips{{color:{MUTED};font-size:12.5px;margin:14px 0 26px;
 font-variant-numeric:tabular-nums}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:26px}}
.kpi{{background:{PANEL};border:1px solid {GRID};border-radius:12px;padding:14px 15px}}
.kt{{color:{INK2};font-size:12px;margin-bottom:8px;min-height:2.6em}}
.kv{{font-size:30px;font-weight:650;letter-spacing:-.02em;color:{OK}}}
.ks{{color:{MUTED};font-size:11.5px;margin-top:6px;min-height:2.6em}}
.card{{background:{PANEL};border:1px solid {GRID};border-radius:12px;
 padding:18px 18px 14px;margin-bottom:18px}}
.card h3{{margin:0 0 3px;font-size:15.5px}}
.csub{{color:{INK2};font-size:12.5px;margin:0 0 12px;max-width:82ch}}
.cnote{{color:{MUTED};font-size:11.5px;margin:8px 0 0}}
.plot{{overflow-x:auto}}
.heatrow{{display:flex;gap:26px;align-items:flex-start;flex-wrap:wrap;overflow-x:auto}}
.heatrow figure{{margin:0}}
.heatrow figcaption{{color:{INK2};font-size:12px;margin-bottom:6px;text-align:center}}
.cbar{{display:flex;flex-direction:column;align-items:center;gap:4px;
 color:{MUTED};font-size:11px;padding-top:22px}}
.cbar .bar{{width:12px;height:150px;border-radius:3px;
 background:linear-gradient(to top,#1a212b,#5aa0f0)}}
.cbar small{{margin-top:2px}}
.checks{{list-style:none;margin:0;padding:0}}
.checks li{{display:grid;grid-template-columns:64px 1fr;gap:6px 12px;
 padding:9px 0;border-top:1px solid {GRID};align-items:baseline}}
.checks li:first-child{{border-top:none}}
.badge{{grid-row:span 2;align-self:center;font-size:11px;font-weight:700;
 letter-spacing:.03em;color:{OK};border:1px solid {OK};border-radius:6px;
 padding:3px 0;text-align:center}}
.checks li.no .badge{{color:{MUTED};border-color:{MUTED}}}
.cn{{font-weight:600;font-size:13.5px}}
.cd{{color:{INK2};font-size:12.5px;font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;font-size:13px;
 font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:8px 10px;border-top:1px solid {GRID}}}
th:first-child,td:first-child{{text-align:left}}
th{{color:{MUTED};font-weight:600;font-size:11.5px;text-transform:uppercase;
 letter-spacing:.04em}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
details{{margin-top:6px}} summary{{cursor:pointer;color:{INK2};font-size:13px}}
.foot{{color:{MUTED};font-size:12px;margin-top:22px;max-width:82ch}}
a{{color:#4f9dff}}
@media(max-width:760px){{.kpis{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="wrap">
<h1>The lead-monetization flywheel</h1>
<p class="lede">A static model trains once and stops. In production you only learn the
outcome of a (lead, offer) you actually <em>send</em> — so sends are training data, and
the model should compound. This simulation runs that loop against the engine’s own
hidden-truth world and pits five allocation policies against each other.</p>
<p class="lede">The takeaway: <strong>the flywheel nearly closes the gap to a perfect
oracle</strong>, and <strong>exploration (Thompson sampling) buys a demonstrably more
accurate model</strong> — no blind spots — which is the insurance that pays off the moment
the offer mix shifts.</p>
<div class="chips">{cfg_chips} · {cfg['n_leads']:,} leads · {cfg['n_offers']} offers · synthetic data</div>

<div class="kpis">{kpi_html}</div>

{_chart_card("Cumulative revenue, round by round",
             "Both learning policies pull away from the frozen model and leave random blasting far below. Oracle (dashed) is the perfect-information ceiling.",
             c1)}

{_chart_card("Share of the oracle ceiling captured",
             "The flywheel policies climb toward ~90% of what a perfect-information oracle would earn; the frozen and random policies plateau well below.",
             c2)}

{_chart_card("How wrong is each model? (mean EV-error over all lead×offer pairs)",
             "Lower is better. The explorer’s model is the most accurate and the frozen model the least — and the gap is stable across rounds.",
             c3,
             "Revenue and accuracy are different axes: greedy can match Thompson on revenue on this stationary world while carrying a worse model — latent risk that surfaces when things change.")}

{heat_html}

<section class="card"><h3>The claims, checked against the numbers above</h3>
<ul class="checks">{checks}</ul></section>

<section class="card"><h3>Full results</h3>
<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
<details><summary>Method notes</summary>
<p class="foot">World = the engine’s hidden conversion function (reused from
<code>generate.py</code>) minus its per-draw noise, so the oracle and regret are exact;
each send draws convert ~ Bernoulli(true mean). Every learning policy starts from the
<em>same</em> cold-start sample and makes the same number of sends per round, so the
comparison is apples-to-apples. Thompson sampling is bootstrapped: an ensemble of models,
each fit on a resample of the accumulated data; each lead acts on one randomly drawn head,
so under-sampled regions (high ensemble disagreement) get explored. Models reuse the
production feature pipeline from <code>recommend.py</code>.</p></details></section>

<p class="foot">Generated by <code>flywheel.py</code>. All data is synthetic — this
demonstrates the learning architecture, not results on real campaign data.</p>
</div></body></html>"""
