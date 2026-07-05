# Lead Monetization App

An MVP single-page app that turns a machine-learning + LLM **lead monetization engine** into an
interactive product. You describe an offer campaign in plain English — _"we signed an auto-insurance
offer paying \$22/lead, who do we send it to and what's the revenue from 500 sends?"_ — and an AI
agent routes the question through the ML model and answers with **grounded numbers**: which leads to
target, projected revenue, and the email/SMS copy to send.

- **React frontend** — an assumptions panel that explains the model and a back-and-forth chat.
- **Flask backend** — serve the built frontend and answer chat queries.
- **AI agent** — deciphers the request, calls the ML tools (never invents numbers), and replies in natural language.
- **MCP dispatch layer** — a real MCP server (simulated ESP + optional live Gmail/Twilio sending)
  that stages campaigns and dispatches them only after in-chat human approval.

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
                                              │  tool-use loop over 4 native tools:      │
                                              │   list_offers · estimate_campaign        │
                                              │   recommend_offers_for_lead · draft_msg  │
                                              │  + 4 campaign tools served over MCP      │
                                              └───────┬──────────────────┬───────────────┘
                                                      ▼                  ▼ MCP (stdio, JSON-RPC)
                  ┌───────────────────────────────────────────┐  ┌──────────────────────────────┐
                  │  ML + LLM engine (unchanged core)          │  │  MCP server (mcp_server.py)  │
                  │   recommend.py  costs.py  llm_layer.py     │  │  stage · dispatch · status   │
                  │   model_store.py → cached model.pkl        │  │  ledger.py → campaigns.db    │
                  └────────────────────────────────────────────┘  │  senders.py → Gmail/Twilio   │
                                                                  │  (sandbox-redirected + capped)│
                                                                  └──────────────────────────────┘
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
│   ├── mcp_server.py         ← MCP server: stage/dispatch campaigns (the "ESP side")
│   ├── mcp_client.py         ← sync bridge: spawns the server, merges its tools
│   ├── ledger.py             ← SQLite outbox: campaigns + per-send status
│   ├── senders.py            ← Gmail SMTP / Twilio transports, sandbox-redirected
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

### Quick start

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > backend/.env
./start_local.sh            # bootstraps venv/deps, builds the SPA, serves http://localhost:5000
```

`./start_local.sh --dev` runs hot-reload dev mode (Vite on :5173 + Flask on :5000);
`./start_local.sh --rebuild` forces a fresh SPA build first. Or do the steps manually:

### 1. Set your API key

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > backend/.env
```

`backend/.env` is gitignored and loaded automatically via `python-dotenv`.

**Optional — live sending.** With no extra config, campaign dispatch runs fully simulated (every
send is recorded in the ledger, nothing leaves the machine). To demo real sends, add to
`backend/.env`:

| Variable | For | Default |
| --- | --- | --- |
| `LIVE_SEND` | opt-in gate for live mode | `0` (all simulated) |
| `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` | live email (app password requires 2FA) | — |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | live SMS | — |
| `DEMO_RECIPIENT_EMAIL`, `DEMO_RECIPIENT_PHONE` | the **only** live destinations (phone must be Twilio-trial-verified) | — |
| `LIVE_SEND_CAP` | how many top-EV sends go live per campaign | `1` (clamped 0–3) |
| `CAMPAIGN_DB_PATH` | ledger location override | `backend/campaigns.db` |

Live mode is **sandbox-redirected**: the send functions physically cannot address anyone but
`DEMO_RECIPIENT_*` (your own inbox/phone), and at most `LIVE_SEND_CAP` messages go live per
campaign — the remainder are simulated. Missing credentials for a channel silently keep that
channel simulated.

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
- **Campaign dispatch:** _"Stage an email campaign for a \$40 insurance quote-request offer to the
  top 200 leads."_ — the agent drafts copy, stages the campaign, and presents a summary (recipients,
  projected revenue, mode, copy). Reply _"yes, send it"_ and it dispatches; ask _"what's the status
  of that campaign?"_ afterward.

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

## The agent (4 native tools + 4 MCP tools)

`agent.py` runs a manual Anthropic tool-use loop (`claude-opus-4-8`, adaptive thinking, iteration cap 8):

- **`list_offers`** — the offer catalog and per-category payout ranges.
- **`estimate_campaign`** _(primary)_ — builds an offer (real or hypothetical), scores all leads, and returns who to send to, the floor, projected revenue, and a budget-lift table (model top-K vs random K). Payout only scales EV, so a described-but-not-in-catalog offer works.
- **`recommend_offers_for_lead`** — best offers for a described lead, ranked by EV.
- **`draft_message`** — generates email/SMS copy for a representative targeted lead via `llm_layer.generate_copy`.

Copy generation stays on the cheaper `claude-haiku-4-5`; the reasoning/orchestration agent is on Opus 4.8.

### Campaign dispatch over MCP

The four campaign tools are not native — they're served by a real **MCP server**
(`mcp_server.py`, Python `mcp` SDK, stdio transport) that the backend spawns as a subprocess and
connects to as an MCP client (`mcp_client.py`). Its tool schemas are fetched over the protocol and
merged into the agent's tool list at runtime:

- **`stage_campaign`** — scores the leads for an offer, resolves the target segment, records it in
  the SQLite ledger with the drafted copy. **Sends nothing.**
- **`dispatch_campaign`** — executes a staged campaign, exactly once (an atomic
  `staged → dispatched` claim in SQLite guards double-sends even across processes).
- **`get_campaign_status`** / **`list_campaigns`** — report from the ledger.

The flow is **two-phase with a human gate**: the agent estimates → drafts copy → stages → presents
the summary and stops; it may dispatch only after the user explicitly approves in a later message.
The approval is prompt-enforced, but dispatch-once is *server*-enforced.

The MCP boundary carries the trust story:

- The **agent process** holds `ANTHROPIC_API_KEY` and never sees a full email/phone — the server
  masks every contact in tool results (`u***7@example.com`).
- The **MCP server** holds the SMTP/Twilio credentials and does the `lead_id → contact` join; it
  never talks to an LLM. In live mode it physically can't address anyone but `DEMO_RECIPIENT_*`.

---

## Notes & deferred work

- **Model persistence** pins `scikit-learn==1.9.0` so `model.pkl` unpickles under the same version; it retrains automatically if the pickle can't load.
- **Performance:** lead scoring is vectorized (one `predict_proba` over all leads per offer) — output is identical to the original per-lead loop, just far faster on the chat hot path.
- **Campaign ledger** (`backend/campaigns.db`) is SQLite in WAL mode — under gunicorn each worker
  spawns its own MCP server subprocess, and SQLite arbitrates the shared outbox. In Docker it lives
  on the container's writable layer (ephemeral across deploys — fine for a demo).
- **Graceful degradation:** if the MCP server fails to start, chat still works with the four native
  tools; the campaign tools simply don't appear.
- **Not built for this MVP** (see `backend/README.md`): token streaming, prompt caching, and a
  production ESP integration (the MCP dispatch layer sends via Gmail/Twilio under a sandbox
  redirect — a real deployment would swap `senders.py` for the company's ESP/SMS platform API).

---

## 1. What does this tool do?

It's a lead monetization engine with a chat interface. A media buyer describes a campaign in plain
English — _"we signed an auto-insurance offer paying \$22/lead, who do we send it to and what's the
revenue from 500 sends?"_ — and an AI agent answers with grounded numbers: which leads to target,
projected revenue, the lift versus random targeting, and drafted email/SMS copy.

Under the hood, a calibrated logistic regression scores every lead–offer pair for conversion
probability, and offers are ranked by expected value (`EV = P(convert) × payout`). Calibration
matters because the output is dollars — a predicted probability has to be a _real_ probability for
the revenue projection to mean anything. The core design principle is that **the model does the math
and the LLM does the language**: the agent never invents a number, it only calls tools (list offers,
estimate a campaign, recommend offers for a lead, draft copy), so every figure in an answer traces
back to the model. It works in both directions — offer → which leads are worth sending to under a
send budget or EV floor, and lead → which offers are worth showing them.

The data (2,000 leads, 15 offers, all outcomes) is synthetic and clearly labeled as such; the
pipeline retrains unchanged on real data.

## 2. Why did I build this one?

The brief says the point of the role is to "build whatever unlocks the most value for the media
buying operation" — and the examples given (video creative generator, ad upload via MCP, landing
page generator) all sit on the **acquisition** side: getting leads in cheaper. That side is already
being actively built.

The business model is buy media → capture leads (cost) → send affiliate offers (revenue). The
**revenue side** — monetizing the leads you already have — is where the margin actually lives, and
nothing in the brief addressed it. Every lead has an acquisition cost; the business only profits if
revenue from that lead exceeds it, and the decision driving that revenue is _"of everything we could
send this lead, what maximizes expected value — and is it even worth a send?"_ That's a decision
currently made by intuition, and it's exactly the kind of decision a calibrated model makes better
than a human at scale.

The role description also says the right person "walks in and starts seeing opportunities I haven't
articulated yet." So instead of building a fourth version of something already on the list, I built
the thing the list was missing — and built it the way I'd argue AI tools should be built for an
ROI-driven business: the ML model owns every number, the LLM owns parsing, orchestration, and copy,
and the agent makes the whole thing usable by a media buyer in plain English.

## 3. What would I build next if this was my full-time job?

I'd close three loops, in order:

**1. Close the data loop — the retraining flywheel.** Right now the model trains once on historical
outcomes. But every campaign the tool recommends generates new labeled data: each send either
converts or doesn't. I'd build the loop where send outcomes flow back in and the model retrains on a
regular cadence — so the tool literally gets smarter every time it's used, on the company's own
campaign data. The two real engineering problems are conversion windows (slow-converting offer types
would otherwise get mislabeled as failures if you retrain too early) and tolerance for
postback/attribution loss. Alongside this, add conversion-history features (prior conversions,
recency, last category) — proven responders convert at multiples of the base rate, and it's likely
the single biggest model lift available.

**2. Close the execution loop — MCP dispatch.** _(Now built in MVP form — see "Campaign dispatch
over MCP" above.)_ An MCP server stages campaigns and dispatches them after an in-chat
human-approval gate, with real sends via Gmail/Twilio under a sandbox redirect. The production
step is swapping the demo transports for the company's actual ESP/SMS platform API — which plugs
directly into the MCP work the team is already doing on the ad-upload side.

**3. Close the business loop — feed monetization back into media buying.** Once the model knows the
expected lifetime revenue of a lead by source, campaign, and creative, you can score _acquisition_
channels by the revenue their leads actually produce — not by cost-per-lead. That turns media buying
from "which source is cheapest" into "which source is most profitable," which is the whole ROI game.

Beyond the loops: send-time/channel/frequency optimization per lead (fatigue modeling), an LLM copy
A/B loop where winning variants feed back into generation, licensed lead enrichment for thin leads,
and a hard eligibility filter that parses plain-text offer rules.
