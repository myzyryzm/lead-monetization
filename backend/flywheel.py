"""
flywheel.py — the learning flywheel + exploration bandit, as a simulation.

The shipped engine is *static*: it trains once on the full `pairs.csv` (every
lead x offer outcome handed to it) and scores. Production never works that way —
you only learn the outcome of a (lead, offer) pair you actually *send*. Those
outcomes become tomorrow's training labels, and the model improves round over
round. That loop is the flywheel.

The subtlety (called out in backend/README.md): the allocation policy shapes its
own training data. A pure-greedy policy stops sending to regions it currently
believes are weak, so it never collects the data that would correct a wrong
belief — it can trap itself. The fix is exploration. This module pits five
allocation policies against each other over many rounds to show it:

    oracle    — sends by TRUE expected value. Theoretical ceiling / regret base.
    thompson  — flywheel + bootstrapped Thompson-sampling exploration.
    greedy    — flywheel, exploit-only (retrain, send model's top-K EV).
    static    — trained once on the cold-start sample, never retrained.
    random    — random sends. Floor / control.

The world is the engine's own hidden-truth conversion function (reused from
generate.py). We deliberately drop generate.py's per-draw Gaussian noise so the
ground-truth mean is well-defined — that makes the Oracle and the regret exact.
Outcomes are still stochastic: each send draws converted ~ Bernoulli(true mean).

Run:
    python flywheel.py                       # defaults, writes CSV + HTML report
    python flywheel.py --rounds 30 --budget 500 --seed 11
    python flywheel.py --reps 3              # average seeds for a stable headline
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data
# Reuse the exact feature spec + vectorized scorer so the sim stays in lockstep
# with the production pipeline (recommend.py) — no re-declaring column lists.
from recommend import CATEGORICAL, NUMERIC, MATCH, _score_leads_vectorized
# Reuse the hidden-world constants from the data generator (import-safe: its
# CSV-writing lives under `if __name__ == "__main__"`).
from generate import SOURCE_QUALITY, COMMITMENT_EFFECT

INTENT_ADD = {"match": 0.12, "related": 0.05, "mismatch": 0.0}
POLICY_ORDER = ["oracle", "thompson", "greedy", "static", "random"]
LEARNERS = ("greedy", "thompson", "static")          # policies that hold a model
FLYWHEEL = ("greedy", "thompson")                    # policies that keep learning


# ----------------------------------------------------------------------
# The world: leads x offers, the hidden true conversion probability, and
# helpers to turn (lead, offer) index pairs into model features.
# ----------------------------------------------------------------------
class World:
    def __init__(self, leads_df, offers_df):
        self.leads = leads_df.reset_index(drop=True)
        self.offers = offers_df.reset_index(drop=True)
        self.n_leads = len(self.leads)
        self.n_offers = len(self.offers)

        # per-lead feature arrays (positional)
        self.L_source = self.leads["source_platform"].to_numpy()
        self.L_intent = self.leads["intent_category"].to_numpy()
        self.L_geo = self.leads["geo_state"].to_numpy()
        self.L_device = self.leads["device"].to_numpy()
        self.L_dss = self.leads["days_since_signup"].to_numpy()
        self.L_opens = self.leads["total_opens"].to_numpy()
        self.L_dslo = self.leads["days_since_last_open"].to_numpy()
        # per-offer feature arrays (positional)
        self.O_cat = self.offers["category"].to_numpy()
        self.O_commit = self.offers["commitment_level"].to_numpy()
        self.O_payout = self.offers["payout"].to_numpy(dtype=float)

        self.intents = list(data.CATEGORIES)
        self.offer_cats = list(data.CATEGORIES)

        # intent_match string for every (lead, offer) — same MATCH matrix the
        # generator and the scorer use.
        self.IM = np.empty((self.n_leads, self.n_offers), dtype=object)
        for j in range(self.n_offers):
            cat = self.O_cat[j]
            self.IM[:, j] = np.array([MATCH[ic][cat] for ic in self.L_intent], dtype=object)

        self.P_true = self._true_p_matrix()
        self.EV_true = self.P_true * self.O_payout[None, :]

    def _true_p_matrix(self):
        """Ground-truth mean conversion prob for every (lead, offer).

        Reuses generate.true_prob's additive terms EXACTLY, minus the per-draw
        Gaussian noise (so the mean — and thus Oracle/regret — is exact).
        """
        d = self.L_dslo
        recency = np.select([d <= 7, d <= 30, d <= 90], [0.05, 0.02, 0.0], default=-0.02)
        opens = np.minimum(self.L_opens, 10) * 0.003
        src = np.array([SOURCE_QUALITY[s] for s in self.L_source])
        s = self.L_dss
        signup = np.select([s <= 14, s > 180], [0.03, -0.02], default=0.0)
        lead_base = 0.02 + recency + opens + src + signup          # (n_leads,)

        P = np.empty((self.n_leads, self.n_offers))
        for j in range(self.n_offers):
            im = self.IM[:, j]
            intent = np.select([im == "match", im == "related"], [0.12, 0.05], default=0.0)
            commit = COMMITMENT_EFFECT[int(self.O_commit[j])]
            P[:, j] = np.clip(lead_base + intent + commit, 0.001, 0.95)
        return P

    def build_features(self, li, oj):
        """(lead idx array, offer idx array) -> feature DataFrame for training."""
        li = np.asarray(li, dtype=int)
        oj = np.asarray(oj, dtype=int)
        return pd.DataFrame({
            "source_platform": self.L_source[li],
            "intent_category": self.L_intent[li],
            "geo_state": self.L_geo[li],
            "device": self.L_device[li],
            "offer_category": self.O_cat[oj],
            "intent_match": self.IM[li, oj],
            "days_since_signup": self.L_dss[li],
            "total_opens": self.L_opens[li],
            "days_since_last_open": self.L_dslo[li],
            "commitment_level": self.O_commit[oj],
        })[CATEGORICAL + NUMERIC]


# ----------------------------------------------------------------------
# Model: same pipeline as recommend.train_model, but fit on ALL accumulated
# rows (no 25% holdout waste) with a cold-start guard.
# ----------------------------------------------------------------------
class _ConstProb:
    """Fallback 'model' for degenerate data (one class / too few rows):
    predict the base rate for everyone. Keeps the flywheel from crashing on
    the earliest, sparsest rounds."""

    def __init__(self, p):
        self.p = float(np.clip(p, 1e-4, 0.95))

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _make_pipeline():
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),   # no class_weight -> honest probs
    ])


def fit_pipeline(X, y):
    """Fit the production feature pipeline on X, y. Falls back to a constant
    base-rate predictor when a fold/bootstrap has a single class or too few rows
    (so early rounds and unlucky bootstraps never crash)."""
    y = np.asarray(y)
    if len(y) < 12 or np.unique(y).size < 2:
        return _ConstProb(y.mean() if len(y) else 0.05)
    pipe = _make_pipeline()
    pipe.fit(X, y)
    return pipe


def prob_matrix(model, world):
    """P(convert) for every (lead, offer) via the reused vectorized scorer."""
    P = np.empty((world.n_leads, world.n_offers))
    for j in range(world.n_offers):
        probs, _ = _score_leads_vectorized(model, world.offers.iloc[j], world.leads)
        P[:, j] = probs
    return P


# ----------------------------------------------------------------------
# Allocation policies. Each returns (lead_idx array, offer_idx array): the K
# sends for the round, at most one offer per lead (frequency cap).
# ----------------------------------------------------------------------
def _topk_from_ev(EV, K):
    """For each lead pick its best offer; send to the K leads with highest best-EV."""
    best_off = EV.argmax(axis=1)
    best_ev = EV[np.arange(EV.shape[0]), best_off]
    top_leads = np.argpartition(-best_ev, K - 1)[:K]
    # order not important for outcomes, but sort for stable/inspectable output
    top_leads = top_leads[np.argsort(-best_ev[top_leads])]
    return top_leads.astype(int), best_off[top_leads].astype(int)


def _select_random(world, K, rng):
    leads = rng.choice(world.n_leads, size=K, replace=False)
    offs = rng.integers(0, world.n_offers, size=K)
    return leads.astype(int), offs.astype(int)


def _select_thompson(ev_stack, K, rng):
    """Bootstrapped Thompson sampling: each lead acts on ONE randomly drawn
    ensemble head (posterior sample). Leads in under-sampled regions have high
    between-head disagreement, so they get explored."""
    B, n, _ = ev_stack.shape
    heads = rng.integers(0, B, size=n)
    sampled = ev_stack[heads, np.arange(n), :]        # (n_leads, n_offers)
    return _topk_from_ev(sampled, K)


# ----------------------------------------------------------------------
# One simulation run (single seed).
# ----------------------------------------------------------------------
def _cell_error_map(model_EV, world):
    """5x5 mean |model_EV - true_EV| per (lead intent, offer category) cell."""
    err = np.abs(model_EV - world.EV_true)
    out = np.zeros((len(world.intents), len(world.offer_cats)))
    for r, intent in enumerate(world.intents):
        lmask = world.L_intent == intent
        for c, cat in enumerate(world.offer_cats):
            omask = world.O_cat == cat
            sub = err[np.ix_(lmask, omask)]
            out[r, c] = float(sub.mean()) if sub.size else 0.0
    return out


def simulate_once(world, cfg, seed):
    rng = np.random.default_rng(seed)
    T, K, B = cfg["rounds"], min(cfg["budget"], world.n_leads), cfg["ensemble"]
    R, warm, postback = cfg["retrain_every"], cfg["warmstart"], cfg["postback"]

    obs = {p: {"li": [], "oj": [], "y": []} for p in FLYWHEEL}
    cells_seen = {p: set() for p in POLICY_ORDER}

    def _observe(pol, li, oj, conv_true):
        """Record postback-attributed labels for a flywheel policy."""
        y_obs = (conv_true * (rng.random(len(li)) < postback)).astype(int)
        obs[pol]["li"].extend(li.tolist())
        obs[pol]["oj"].extend(oj.tolist())
        obs[pol]["y"].extend(y_obs.tolist())

    # --- cold start: one shared pool of random observations seeds every learner
    wl = rng.integers(0, world.n_leads, size=warm)
    wo = rng.integers(0, world.n_offers, size=warm)
    w_conv = (rng.random(warm) < world.P_true[wl, wo]).astype(int)
    w_y = (w_conv * (rng.random(warm) < postback)).astype(int)
    for p in LEARNERS:
        obs.setdefault(p, {"li": [], "oj": [], "y": []})
        obs[p]["li"].extend(wl.tolist())
        obs[p]["oj"].extend(wo.tolist())
        obs[p]["y"].extend(w_y.tolist())
        for i, j in zip(wl.tolist(), wo.tolist()):
            cells_seen[p].add((world.L_intent[i], world.O_cat[j]))

    def _fit(pol):
        X = world.build_features(obs[pol]["li"], obs[pol]["oj"])
        return fit_pipeline(X, np.array(obs[pol]["y"]))

    def _fit_ensemble(pol):
        li = np.array(obs[pol]["li"]); oj = np.array(obs[pol]["oj"]); y = np.array(obs[pol]["y"])
        n = len(y)
        members = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)          # bootstrap resample
            members.append(fit_pipeline(world.build_features(li[idx], oj[idx]), y[idx]))
        return members

    static_model = _fit("static")                     # trained once, frozen
    static_P = prob_matrix(static_model, world)
    static_EV = static_P * world.O_payout[None, :]
    static_err = float(np.abs(static_EV - world.EV_true).mean())

    greedy_model = _fit("greedy")
    thompson_members = _fit_ensemble("thompson")

    series = {p: {k: [] for k in
                  ("cum_revenue", "round_revenue", "conversions", "coverage", "ev_error")}
              for p in POLICY_ORDER}
    cum_rev = {p: 0.0 for p in POLICY_ORDER}
    cum_conv = {p: 0 for p in POLICY_ORDER}
    heat = {"greedy": None, "thompson": None}

    for t in range(1, T + 1):
        # score once per round; reuse for both selection and error metrics
        greedy_P = prob_matrix(greedy_model, world)
        greedy_EV = greedy_P * world.O_payout[None, :]
        prob_stack = np.stack([prob_matrix(m, world) for m in thompson_members])
        ev_stack = prob_stack * world.O_payout[None, None, :]
        thom_meanEV = ev_stack.mean(axis=0)

        picks = {
            "oracle":   _topk_from_ev(world.EV_true, K),
            "greedy":   _topk_from_ev(greedy_EV, K),
            "static":   _topk_from_ev(static_EV, K),
            "thompson": _select_thompson(ev_stack, K, rng),
            "random":   _select_random(world, K, rng),
        }
        round_err = {
            "greedy": float(np.abs(greedy_EV - world.EV_true).mean()),
            "thompson": float(np.abs(thom_meanEV - world.EV_true).mean()),
            "static": static_err,
        }

        for p in POLICY_ORDER:
            li, oj = picks[p]
            conv = (rng.random(len(li)) < world.P_true[li, oj]).astype(int)
            rev = float((world.O_payout[oj] * conv).sum())
            cum_rev[p] += rev
            cum_conv[p] += int(conv.sum())
            for i, j in zip(li.tolist(), oj.tolist()):
                cells_seen[p].add((world.L_intent[i], world.O_cat[j]))
            if p in FLYWHEEL:
                _observe(p, li, oj, conv)
            series[p]["cum_revenue"].append(cum_rev[p])
            series[p]["round_revenue"].append(rev)
            series[p]["conversions"].append(cum_conv[p])
            series[p]["coverage"].append(len(cells_seen[p]))
            series[p]["ev_error"].append(round_err.get(p))

        if t == T:                                    # capture final blind-spot maps
            heat["greedy"] = _cell_error_map(greedy_EV, world)
            heat["thompson"] = _cell_error_map(thom_meanEV, world)

        if t % R == 0:                                # batched retrain
            greedy_model = _fit("greedy")
            thompson_members = _fit_ensemble("thompson")

    return {"series": series, "cum_rev": cum_rev, "cum_conv": cum_conv, "heat": heat}


# ----------------------------------------------------------------------
# Public entry point: run reps, average, package a JSON-safe results dict.
# ----------------------------------------------------------------------
_WORLD = None


def get_world():
    global _WORLD
    if _WORLD is None:
        _WORLD = World(data.get_leads(), data.get_offers())
    return _WORLD


def _avg(list_of_lists):
    return np.mean(np.array(list_of_lists, dtype=float), axis=0).tolist()


def simulate(cfg):
    """Run the simulation and return a JSON-safe results dict (used by the CLI,
    the HTML report, and the /api/flywheel route)."""
    cfg = _normalize_cfg(cfg)
    world = get_world()
    runs = [simulate_once(world, cfg, cfg["seed"] + r) for r in range(cfg["reps"])]

    rounds = list(range(1, cfg["rounds"] + 1))
    metrics = ("cum_revenue", "round_revenue", "conversions", "coverage")
    series = {}
    for p in POLICY_ORDER:
        s = {m: _avg([run["series"][p][m] for run in runs]) for m in metrics}
        # ev_error is None for oracle/random; average only where numeric
        ev = runs[0]["series"][p]["ev_error"]
        s["ev_error"] = (None if ev[0] is None
                         else _avg([run["series"][p]["ev_error"] for run in runs]))
        series[p] = s

    oracle_cum = series["oracle"]["cum_revenue"]
    for p in POLICY_ORDER:
        series[p]["pct_of_oracle"] = [
            (100.0 * c / o) if o > 0 else 0.0
            for c, o in zip(series[p]["cum_revenue"], oracle_cum)
        ]

    summary = {}
    for p in POLICY_ORDER:
        ev = series[p]["ev_error"]
        summary[p] = {
            "total_revenue": round(series[p]["cum_revenue"][-1], 2),
            "total_conversions": int(round(series[p]["conversions"][-1])),
            "pct_of_oracle": round(series[p]["pct_of_oracle"][-1], 1),
            "final_coverage": int(round(series[p]["coverage"][-1])),
            "mean_ev_error": (None if ev is None else round(float(np.mean(ev)), 4)),
        }

    # heatmaps: average cell-error across reps, share one color scale
    heat = {}
    for p in ("greedy", "thompson"):
        heat[p] = np.mean([run["heat"][p] for run in runs], axis=0)
    hmax = float(max(heat["greedy"].max(), heat["thompson"].max()))
    heatmaps = {
        "intents": world.intents,
        "offer_categories": world.offer_cats,
        "greedy": [[round(v, 3) for v in row] for row in heat["greedy"].tolist()],
        "thompson": [[round(v, 3) for v in row] for row in heat["thompson"].tolist()],
        "max": round(hmax, 3),
    }

    return {
        "config": {**{k: cfg[k] for k in
                       ("rounds", "budget", "ensemble", "warmstart",
                        "retrain_every", "postback", "seed", "reps")},
                   "n_leads": world.n_leads, "n_offers": world.n_offers},
        "policies": POLICY_ORDER,
        "rounds": rounds,
        "series": series,
        "summary": summary,
        "heatmaps": heatmaps,
        "assertions": _thesis_assertions(summary),
    }


def _thesis_assertions(summary):
    """The claims the demo makes, checked against the numbers it produced.

    Only the seed-robust claims are asserted. Exploration's effect on *revenue*
    is reported transparently (it's a wash on this stationary, well-featured
    world — see the last check's detail), because the honest payoff of
    exploration here is a more accurate model, not guaranteed extra revenue."""
    rev = {p: summary[p]["total_revenue"] for p in POLICY_ORDER}
    pct = {p: summary[p]["pct_of_oracle"] for p in POLICY_ORDER}
    err = {p: summary[p]["mean_ev_error"] for p in ("thompson", "greedy", "static")}
    tol = 1.0 + 1e-9
    checks = [
        ("Flywheel beats a frozen model  (greedy ≥ static)",
         rev["greedy"] >= rev["static"] / tol,
         f"${rev['greedy']:,.0f} vs ${rev['static']:,.0f}  "
         f"(+{100 * (rev['greedy'] / rev['static'] - 1):.0f}%)"),
        ("Learning beats blind sending  (every learner ≥ random)",
         min(rev["greedy"], rev["thompson"], rev["static"]) >= rev["random"] / tol,
         f"min learner ${min(rev['greedy'], rev['thompson'], rev['static']):,.0f} "
         f"vs random ${rev['random']:,.0f}"),
        ("Oracle is the ceiling  (oracle ≥ all)",
         rev["oracle"] >= max(rev[p] for p in POLICY_ORDER if p != "oracle") / tol,
         f"${rev['oracle']:,.0f}"),
        ("Flywheel nearly closes the gap to oracle  (both ≥ 80%)",
         min(pct["greedy"], pct["thompson"]) >= 80.0,
         f"thompson {pct['thompson']:.0f}% · greedy {pct['greedy']:.0f}% of oracle"),
        ("Exploration yields the most accurate model  (thompson < greedy < static error)",
         err["thompson"] <= err["greedy"] * tol <= err["static"] * tol * tol,
         f"EV-error thompson {err['thompson']} < greedy {err['greedy']} < static {err['static']}"),
        ("(reported) Exploration's revenue effect on a stationary world",
         True,
         f"thompson ${rev['thompson']:,.0f} vs greedy ${rev['greedy']:,.0f}  "
         f"(Δ ${rev['thompson'] - rev['greedy']:+,.0f}) — a wash; its payoff is model accuracy above"),
    ]
    return [{"name": n, "pass": bool(ok), "detail": d} for n, ok, d in checks]


# ----------------------------------------------------------------------
# Config normalization + safe caps (shared by CLI and the API route).
# ----------------------------------------------------------------------
CAPS = {"rounds": (1, 60), "budget": (10, 1200), "ensemble": (2, 16),
        "warmstart": (0, 5000), "retrain_every": (1, 10), "reps": (1, 5)}
DEFAULTS = {"rounds": 24, "budget": 400, "ensemble": 8, "warmstart": 300,
            "retrain_every": 1, "postback": 1.0, "seed": 7, "reps": 1}


def _normalize_cfg(cfg):
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    for k, (lo, hi) in CAPS.items():
        out[k] = int(min(max(int(out[k]), lo), hi))
    out["postback"] = float(min(max(float(out["postback"]), 0.1), 1.0))
    out["seed"] = int(out["seed"])
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _write_csv(res, path):
    rows = []
    for p in res["policies"]:
        s = res["series"][p]
        for i, rnd in enumerate(res["rounds"]):
            rows.append({
                "round": rnd, "policy": p,
                "cum_revenue": round(s["cum_revenue"][i], 2),
                "round_revenue": round(s["round_revenue"][i], 2),
                "conversions": round(s["conversions"][i], 1),
                "pct_of_oracle": round(s["pct_of_oracle"][i], 2),
                "coverage": round(s["coverage"][i], 2),
                "ev_error": (None if s["ev_error"] is None else round(s["ev_error"][i], 4)),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _print_summary(res):
    c = res["config"]
    print(f"\nFlywheel simulation — {c['rounds']} rounds x {c['budget']} sends "
          f"| ensemble {c['ensemble']} | warm-start {c['warmstart']} | "
          f"retrain every {c['retrain_every']} | postback {c['postback']} | "
          f"seed {c['seed']} | reps {c['reps']}")
    print("=" * 78)
    print(f"{'policy':<10}{'revenue':>13}{'convs':>9}{'% oracle':>11}"
          f"{'coverage':>11}{'EV-error':>11}")
    print("-" * 78)
    for p in res["policies"]:
        s = res["summary"][p]
        err = "  n/a" if s["mean_ev_error"] is None else f"{s['mean_ev_error']:.4f}"
        print(f"{p:<10}{'$'+format(s['total_revenue'], ',.0f'):>13}"
              f"{s['total_conversions']:>9}{s['pct_of_oracle']:>10.1f}%"
              f"{str(s['final_coverage'])+'/25':>11}{err:>11}")
    print("-" * 78)
    print("Thesis checks:")
    for a in res["assertions"]:
        print(f"  [{'PASS' if a['pass'] else 'FAIL'}] {a['name']}  —  {a['detail']}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Learning-flywheel + bandit simulation.")
    ap.add_argument("--rounds", type=int, default=DEFAULTS["rounds"])
    ap.add_argument("--budget", type=int, default=DEFAULTS["budget"], help="sends per round (K)")
    ap.add_argument("--ensemble", type=int, default=DEFAULTS["ensemble"], help="Thompson bootstrap heads (B)")
    ap.add_argument("--warmstart", type=int, default=DEFAULTS["warmstart"], help="shared cold-start observations")
    ap.add_argument("--retrain-every", type=int, default=DEFAULTS["retrain_every"], dest="retrain_every")
    ap.add_argument("--postback", type=float, default=DEFAULTS["postback"], help="attribution rate (1.0 = no loss)")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--reps", type=int, default=DEFAULTS["reps"], help="average this many seeds")
    ap.add_argument("--csv", default="flywheel_results.csv")
    ap.add_argument("--html", default="flywheel_report.html")
    ap.add_argument("--no-report", action="store_true", help="skip the HTML report")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = {k: getattr(args, k) for k in
           ("rounds", "budget", "ensemble", "warmstart", "retrain_every",
            "postback", "seed", "reps")}
    res = simulate(cfg)

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
    _write_csv(res, csv_path)
    _print_summary(res)
    print(f"\nwrote {csv_path}")

    if not args.no_report:
        from flywheel_report import render
        html_path = args.html if os.path.isabs(args.html) else os.path.join(here, args.html)
        with open(html_path, "w") as f:
            f.write(render(res))
        print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
