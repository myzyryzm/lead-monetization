# Lead Monetization App

An MVP single-page app that turns a machine-learning + LLM **lead monetization engine** into an
interactive product. You describe an offer campaign in plain English — _"we signed an auto-insurance
offer paying \$22/lead, who do we send it to and what's the revenue from 500 sends?"_ — and an AI
agent routes the question through the ML model and answers with **grounded numbers**: which leads to
target, projected revenue, and the email/SMS copy to send.

- **React frontend** — an assumptions panel that explains the model and a back-and-forth chat.
- **Flask backend** — serve the built frontend and answer chat queries.
- **AI agent** — deciphers the request, calls the ML tools (never invents numbers), and replies in natural language.

> **Note:** All data (2000 leads, 15 offers, every conversion outcome) is **randomly generated**.
> This is an MVP demonstrating the agentic integration, not a system trained on real data. See
> [`backend/README.md`](backend/README.md) for the modeling rationale and roadmap to production.

---

## Architecture

```
┌────────────────────┐        POST /api/chat         ┌──────────────────────────────┐
│  React SPA          │  {messages:[{role,content}]}  │  Flask (app.py)              │
│  • AssumptionsPanel │ ───────────────────────────►  │  • serves the built SPA      │
│  • Chat             │ ◄───────────────────────────  │  • GET /api/assumptions      │
└────────────────────┘   {reply, messages}            │  • POST /api/chat            │
                                                        └──────────────┬───────────────┘
                                                                       │ run_agent(messages)
                                                                       ▼
                                              ┌────────────────────────────────────────┐
                                              │  Agent (agent.py) — claude-opus-4-8      │
                                              │  tool-use loop over 4 tools:             │
                                              │   list_offers · estimate_campaign        │
                                              │   recommend_offers_for_lead · draft_msg  │
                                              └───────────────┬──────────────────────────┘
                                                              ▼
                          ┌───────────────────────────────────────────────────────────┐
                          │  ML + LLM engine (unchanged core)                          │
                          │   recommend.py  costs.py  llm_layer.py                     │
                          │   model_store.py → cached, persisted model.pkl             │
                          └───────────────────────────────────────────────────────────┘
```

**The core design principle** (from the engine): _the model does the math; the LLM does the language._
The agent never estimates a probability, revenue figure, or lead count itself — it always calls a tool,
and every number in an answer traces back to the calibrated model.

---

## Project layout

```
its-media-today/
├── README.md                 ← you are here (whole-app overview)
├── .gitignore
├── backend/
│   ├── README.md             ← the ML/LLM engine deep-dive (modeling decisions, roadmap)
│   ├── app.py                ← Flask entry point: serves SPA + /api routes
│   ├── agent.py              ← the AI agent (Anthropic tool-use loop, 4 tools)
│   ├── data.py               ← absolute-path CSV access + catalog_summary()
│   ├── model_store.py        ← train-once / persist to model.pkl (joblib)
│   ├── recommend.py          ← lead → ranked offers by EV  (+ vectorized scoring)
│   ├── costs.py              ← offer → who to send to, under a send budget/EV floor
│   ├── llm_layer.py          ← email/SMS copy generation (claude-haiku-4-5 + fallback)
│   ├── generate.py, train.py ← synthetic-data generation & standalone model training
│   ├── leads.csv, offers.csv, pairs.csv   ← generated data
│   ├── requirements.txt
│   └── .env                  ← ANTHROPIC_API_KEY  (gitignored — see security note)
└── frontend/
    ├── index.html
    ├── package.json, vite.config.js
    ├── src/
    │   ├── main.jsx, App.jsx
    │   ├── api.js             ← fetchAssumptions() / sendChat()
    │   ├── markdown.js        ← sanitized (escape-first) markdown renderer
    │   ├── styles.css
    │   └── components/
    │       ├── AssumptionsPanel.jsx
    │       └── Chat.jsx
    └── dist/                 ← `npm run build` output, served by Flask (gitignored)
```

---

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+
- An `ANTHROPIC_API_KEY`

### 1. Set your API key

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > backend/.env
```

`backend/.env` is gitignored and loaded automatically via `python-dotenv`.

### 2. Install & build

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# frontend (build the SPA that Flask will serve)
cd ../frontend
npm install
npm run build
```

### 3. Run

```bash
cd ../backend
python app.py          # http://localhost:5000
```

On startup the server loads the data and trains the model **once**, persisting it to `model.pkl`
(subsequent runs load the pickle — no retrain). Open **http://localhost:5000** and start chatting.

### Dev mode (hot-reload frontend)

Run Vite and Flask side by side — Vite proxies `/api` to Flask:

```bash
# terminal 1
cd backend && source .venv/bin/activate && python app.py

# terminal 2
cd frontend && npm run dev        # http://localhost:5173
```

---

## How to use it

The chat handles both directions of the monetization decision. Try:

- **Campaign estimation (primary):** _"We signed an auto-insurance offer at \$22/lead — who do we send to and what's the projected revenue from 500 sends?"_
- **Reverse lookup:** _"Best offers for a finance lead from Google search who last opened 5 days ago?"_
- **Copy drafting:** _"Draft an email and SMS for a \$30 health paid-trial offer."_
- **Budget/floor targeting:** _"For a \$6 sweepstakes email-submit offer, who clears a \$0.50/lead EV floor?"_

The agent states the **assumptions it made** (the assumed `commitment_level` and channel) and cites
tool-derived figures — e.g. _"~\$1,721 from 500 email sends, 1.83× a random 500."_

---

## HTTP API

| Method | Route              | Purpose                                                                                                                                                                                                                                                                                        |
| ------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/api/assumptions` | Catalog summary (categories, offers, payout ranges, commitment ladder) + prose disclaimers. Powers the assumptions panel.                                                                                                                                                                      |
| `POST` | `/api/chat`        | Body `{ "messages": [{ "role": "user", "content": "..." }] }`. Runs the agent, returns `{ reply, messages }` where `messages` is the full plain-text history.                                                                                                                                  |
| `*`    | `/<path>`          | Serves the built SPA (real file if it exists, else `index.html` for client routing).                                                                                                                                                                                                           |

```bash
curl -X POST localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"We signed an auto-insurance offer paying $22/lead. Revenue from 500 sends?"}]}'
```

`/api/chat` returns **plain-text** turns only — the agent's internal thinking/tool-use blocks live
inside a single request and are never round-tripped to the client.

---

## The agent (4 tools)

`agent.py` runs a manual Anthropic tool-use loop (`claude-opus-4-8`, adaptive thinking, iteration cap 8):

- **`list_offers`** — the offer catalog and per-category payout ranges.
- **`estimate_campaign`** _(primary)_ — builds an offer (real or hypothetical), scores all leads, and returns who to send to, the floor, projected revenue, and a budget-lift table (model top-K vs random K). Payout only scales EV, so a described-but-not-in-catalog offer works.
- **`recommend_offers_for_lead`** — best offers for a described lead, ranked by EV.
- **`draft_message`** — generates email/SMS copy for a representative targeted lead via `llm_layer.generate_copy`.

Copy generation stays on the cheaper `claude-haiku-4-5`; the reasoning/orchestration agent is on Opus 4.8.

---

## Notes & deferred work

- **Model persistence** pins `scikit-learn==1.9.0` so `model.pkl` unpickles under the same version; it retrains automatically if the pickle can't load.
- **Performance:** lead scoring is vectorized (one `predict_proba` over all leads per offer) — output is identical to the original per-lead loop, just far faster on the chat hot path.
- **Not built for this MVP** (see `backend/README.md`): token streaming, prompt caching, a production WSGI server (gunicorn/waitress), and connecting the agent to a real ESP/SMS platform for dispatch.
