# Lead Monetization Engine

An ML + LLM system that decides **which affiliate offer to send to which lead, and what it's worth** — built to demonstrate an approach to It's Today Media's monetization problem.

---

## Why this problem

The brief asks for "whatever unlocks the most value for our media buying operation" without naming a specific pain point. So the first task was to find the highest-leverage, least-obvious problem.

The business is: buy media to capture email/SMS leads (a cost), then monetize those leads by sending affiliate offers (the revenue). Most of the named build examples — video creative, ad upload, landing pages — optimize the **acquisition** (spend) side. That side is already being worked on, and it's crowded.

The revenue side — **monetizing leads you already have** — is where the actual margin lives and where I'd expect the least competition for ideas. Every lead has an acquisition cost; the business only profits if lifetime revenue from that lead exceeds it. The decision that drives that revenue is: *of all the offers we could send this lead, which one maximizes expected revenue, and is it even worth a send?*

That's the problem this system addresses. It deliberately does **not** touch acquisition (ad → click → landing page → lead capture); the brief makes clear that landing-page/lead-capture tooling already exists. This system starts at "the lead exists" and owns the monetization decision.

---

## What it does

Given a lead and a catalog of offers, the system:

1. **Scores** each lead–offer pair for conversion probability (calibrated ML model).
2. **Ranks** offers by expected value: `EV = P(convert) × payout`.
3. **Targets**: for a given offer, decides which leads are worth sending to under a send budget / EV floor — and explains who is excluded and why.
4. **Generates** the email + SMS copy for the chosen sends (LLM).

The intended interface is an **agent layer**: a natural-language request ("we just signed an auto-insurance offer paying $22 — who do we send it to?") orchestrates these tools and returns the target segment, projected revenue, and drafted copy.

---

## Architecture: the division of labor

The core design principle is **the model does the math; the LLM does the language.** LLMs are poor at calibrated numerical prediction, so they are never asked to decide who converts or how to allocate. That is the model's job. The LLM handles what it's good at: parsing messy input into structure, reasoning where there's no data, generating copy, and orchestrating tools.

```
lead ── intent_match ──┐
                       ├──► calibrated model ──► P(convert)
offer ── commitment ───┘                          │
                                    EV = P × payout │   ◄── payout enters HERE only
                                                   ▼
                            rank / apply send threshold / budget
                                                   ▼
                              LLM: rationale + email + SMS copy
```

| Component | File | Role |
|---|---|---|
| Synthetic data generator | `generate.py` | Produces realistic leads, offers, and lead–offer outcome pairs |
| Model training + inspection | `train.py` | Logistic regression, metrics, weight inspection |
| Model comparison | `compare.py` | LR vs LightGBM; interaction-recovery experiment |
| Calibration | `calibrate.py` | Makes probabilities honest so EV is real dollars |
| Recommendation tool | `recommend.py` | Lead → ranked offers by EV (the core engine) |
| Cost / targeting tool | `costs.py` | Offer → which leads to send to, under budget/floor |
| LLM layer | `llm_layer.py` | Rationale + email/SMS copy generation |

---

## Modeling decisions (and why)

**Logistic regression, not a neural net.** This is tabular data with a handful of columns. The right tools are linear models and gradient-boosted trees, not deep learning. LR was chosen for the core engine because it's interpretable — you can read the learned weights and confirm they match reality — and because it actually outperformed LightGBM on this data (see below).

**Calibration matters because the tool predicts dollars.** Class-imbalanced training (~6% conversion) tempts you to use balanced class weights, which improve ranking but **destroy probability calibration** (the model output 0.71 when the true rate was 0.13). Since `EV = P × payout` is meaningless unless `P` is a real probability, the model is wrapped in isotonic calibration. Result: predicted probabilities match true conversion rates within ~1 point, with ranking (AUC 0.74) fully preserved.

**Payout is excluded from the model.** The consumer never sees payout — it can't causally influence whether they convert. So payout is excluded from `P(convert)` and applied only in the EV ranking. Conversion is driven by factors the lead actually experiences: how well the offer matches their interest (`intent_match`) and how much effort it demands (`commitment_level`).

**`intent_match` is an engineered interaction feature.** Conversion depends on the *pairing* of lead intent and offer category, which a linear model can't derive on its own. `compare.py` shows LightGBM can rediscover this interaction from raw columns via nested splits (it loses only ~0.02 AUC without the hand-built feature) — a demonstration of why trees and linear models differ.

**The send decision is budget-constrained, not pure-EV.** With near-zero per-send cost, raw expected-value math says "blast everyone" — every send has positive EV. The reason you *don't* blast is fatigue / sender-reputation damage, which is real but not cleanly quantifiable in dollars. So targeting is framed as **best use of a limited send budget**: given a budget of K sends, the model's top-K leads capture ~2.3x the revenue of a random K. Fatigue is modeled operationally as a frequency cap (what real operators enforce), not a fabricated per-send dollar cost.

---

## On the data

**The data is synthetic, and that's stated plainly.** I don't have It's Today Media's data, so `generate.py` models realistic affiliate dynamics (low base conversion, intent-match driving conversion, source-quality effects, commitment-level effort, noise) and produces lead–offer outcomes from a hidden probability function. The model never sees that function — it recovers the structure from outcomes alone, which is verified by inspecting that the learned weights point the directions the dynamics were planted.

On real data the pipeline retrains unchanged. The one honest caveat that carries to production: the conversion label is defined by **received advertiser postbacks**, which under-count true conversions due to tracking loss and advertiser under-reporting — so the model learns a slightly conservative estimate of true conversion.

---

## Running it

```bash
pip install -r requirements.txt

python generate.py     # creates leads.csv, offers.csv, pairs.csv
python train.py        # trains + inspects the model
python recommend.py    # demo: lead → ranked offers by EV
python costs.py        # demo: offer → who to send to, budget lift

# LLM copy generation (needs your key):
export ANTHROPIC_API_KEY=sk-ant-...
python llm_layer.py
```

---

## The learning flywheel (built)

`flywheel.py` turns the static engine into the loop that actually compounds in production:
you only learn the outcome of a (lead, offer) you **send**, so sends are training data and the
model should improve round over round. It reuses the engine's own hidden-truth world (from
`generate.py`, minus the per-draw noise so the oracle/regret are exact) and the production feature
pipeline (from `recommend.py`), then races five allocation policies over many rounds:

| policy | what it does |
|---|---|
| **oracle** | sends by *true* EV — the perfect-information ceiling / regret baseline |
| **thompson** | flywheel + **bootstrapped Thompson-sampling** exploration (an ensemble of models on bootstrap resamples; each lead acts on one randomly drawn head, so under-sampled regions get explored) |
| **greedy** | flywheel, exploit-only (retrain, send the model's top-K EV) |
| **static** | trained once on the cold-start sample, never retrained — isolates "no flywheel" |
| **random** | random sends — the floor |

**What it shows** (robust across seeds): the flywheel captures **~90% of the oracle's revenue**,
beating the frozen model by ~40% and random blasting by ~4–5×; and **exploration buys a
demonstrably more accurate model** (lowest EV-error, no blind spots) — the honest finding is that on
a *stationary* world greedy stays revenue-competitive because a feature-generalizing linear model
transfers information across leads, so exploration's payoff here is model accuracy (insurance against
drift), visualized as a per-segment blind-spot heatmap where greedy develops hot cells the explorer
does not.

```bash
python flywheel.py                      # 24 rounds; writes flywheel_results.csv + flywheel_report.html
python flywheel.py --reps 3 --seed 11   # average seeds for a stable headline
python flywheel.py --rounds 30 --budget 500 --ensemble 10 --retrain-every 2 --postback 0.85
```

It also runs live in the app — the **Learning flywheel** tab (`POST /api/flywheel`) runs the sim
server-side and animates the same charts in-browser. That tab is **behind a feature flag and off by
default**: it shows a "coming soon" explanation until you start the server with
`ENABLE_FLYWHEEL=true` (the route returns 503 and `/api/config` reports `flywheel_enabled: false`
while disabled). The CLI above always works regardless of the flag. See the whole-app
[`README.md`](../README.md) for the architecture.

## What's next (production design)

Things deliberately left as roadmap rather than built, because a working core plus a clear roadmap beats a half-built everything:

- **Agent orchestration layer** — natural-language interface over the tools (`recommend`, `target_offer`, `budget_lift`, `generate_copy`). The agent earns its place only on open-ended requests where the tool sequence genuinely varies; deterministic steps stay deterministic.
- **Conversion history as features** — prior-conversion count, recency, last category. Proven responders convert far more; this is likely the single biggest model-lift available, though the data is noisy (postback-dependent).
- **The learning flywheel** — ✅ **built** as a simulation (`flywheel.py`) — see [The learning flywheel](#the-learning-flywheel-built) below. Sends generate new labeled outcomes that retrain the model; the sim demonstrates **exploration** (Thompson sampling, so the policy doesn't trap its early assumptions), **batched** retraining, and postback/attribution loss. A conversion **window** per offer type (slow converters mislabeled as negatives) remains a production concern the sim notes but doesn't yet model.
- **Execution via MCP** — connect to the ESP/SMS platform so the agent dispatches campaigns directly, with a human-approval gate before autonomous sends.
- **Lead enrichment** — append demographics/interests to thin leads via a **licensed** enrichment vendor (not DIY social scraping, which carries ToS and privacy exposure).
- **Eligibility filter** — hard end-stage gate (LLM parses plain-text offer rules), applied after scoring, on the fields actually collected.
- **Vectorized scoring** — current scoring loops per lead (fine for the demo); production builds one feature matrix and scores in a single pass.
