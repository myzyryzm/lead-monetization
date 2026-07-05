"""
The agent layer: a natural-language interface over the ML tools.

Division of labor (the whole architecture in one sentence):
  the MODEL does the math (P(convert), EV, ranking, who-to-send);
  the LLM parses messy input, orchestrates the tools, and writes the answer.

`run_agent(messages)` runs an Anthropic tool-use loop over the four tools below
and returns the final natural-language reply plus the updated PLAIN-text history
(the internal thinking/tool_use/tool_result blocks live only inside one request
and are never sent back to the client).
"""

import json

from anthropic import Anthropic

import data
import costs
from model_store import get_model
from recommend import recommend, MATCH
from llm_layer import generate_copy

MODEL = "claude-opus-4-8"
MAX_ITERS = 8          # hard cap on tool-use round-trips per turn
MAX_TOKENS = 4096

_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _client


SYSTEM_PROMPT = """You are the monetization assistant for "It's Today Media", a \
company that buys media to capture email/SMS leads and then earns revenue by \
sending those leads affiliate offers. You help a media buyer decide WHICH leads \
to send an offer to, estimate the REVENUE, and draft the outreach copy.

THE DATA (an MVP on synthetic data — say so if asked about real-world accuracy):
- 2000 synthetic leads. Each lead has an `intent_category` (what they signed up \
interested in), an acquisition source, device/geo, and engagement/recency signals.
- 5 offer categories: finance, insurance, health, sweepstakes, home_services. \
Each offer has a `payout` (what we earn per conversion) and a `commitment_level`.

HOW THE MODEL WORKS (this is the math — you never do it yourself):
- A calibrated model predicts P(convert) for a lead-offer pair from intent match, \
commitment level, engagement/recency, and source quality. Expected value is \
`EV = P(convert) x payout`.
- `payout` is NOT a conversion driver (the consumer never sees it) — it only \
scales EV. So a described offer that isn't in the catalog works fine.
- `commitment_level` IS a driver (higher effort -> lower conversion): \
1=email_submit, 2=quote_request, 3=long_form, 4=paid_trial, 5=purchase.
- The send decision is BUDGET-CONSTRAINED (you can't blast everyone — fatigue and \
sender-reputation risk). Value is framed as the best use of a K-send budget: the \
model's top-K leads vs a random K. "Revenue from K sends" = the model's top-K \
revenue for that K.

YOUR RULES:
- The model does the math. NEVER invent probabilities, EV, counts, revenue, or \
lift — ALWAYS call a tool and cite its numbers.
- When the user describes an offer, extract the `category` and `payout`, and its \
specific product name (e.g. "auto insurance") — pass that to `draft_message` as \
`offer_name` so the copy is framed around the real product, not the generic category. \
Infer `commitment_level` from their words using the ladder above; if it's unstated, the \
tool defaults to 2 (quote_request) — either way, STATE which level you assumed.
- Default the channel to email and state it. If the user names an EV floor \
("at least $X/lead") pass it as `min_ev`. If they name a send budget, pass \
`budget_k` and report that budget's revenue and its lift vs random.
- Be concise and concrete. Lead with the answer (who to send to / the revenue), \
then a short why. Use the leads' real numbers, not round guesses.

Use `list_offers` to ground yourself in the catalog when the user names an offer \
by title or asks what's available."""


# ---------------------------------------------------------------------------
# Tool implementations (each returns a JSON-serializable dict; an "error" key
# signals failure back to the model).
# ---------------------------------------------------------------------------

def _sample_lead(intent_category: str) -> dict:
    """A deterministic representative lead of the given intent."""
    leads = data.get_leads()
    subset = leads[leads.intent_category == intent_category]
    if len(subset) == 0:
        subset = leads
    return subset.iloc[0].to_dict()


def _build_offer(category, payout, commitment_level=None, offer_name=None) -> dict:
    level = int(commitment_level) if commitment_level else 2
    return {
        "offer_name": offer_name or f"{category.replace('_', ' ').title()} offer",
        "category": category,
        "commitment_level": level,
        "payout": float(payout),
    }


def tool_list_offers(category: str = None) -> dict:
    summary = data.catalog_summary()
    if category:
        summary = {**summary,
                   "categories": [c for c in summary["categories"]
                                  if c["category"] == category]}
    return summary


def tool_estimate_campaign(category, payout, commitment_level=None,
                           channel="email", min_ev=None, budget_k=None) -> dict:
    offer = _build_offer(category, payout, commitment_level)
    model = get_model()
    leads = data.get_leads()

    result = costs.target_offer(model, offer, leads,
                                channel=channel or "email", min_ev=min_ev)

    all_ev = ([r["expected_value"] for r in result["sendable"]]
              + [r["expected_value"] for r in result["excluded"]])
    mean_ev_all = sum(all_ev) / len(all_ev) if all_ev else 0.0

    n_sendable = result["n_sendable"]
    ks = [k for k in (100, 300, 500, 1000) if k <= n_sendable]
    if budget_k:
        ks.append(int(budget_k))
    ks = sorted({k for k in ks if k > 0})
    if not ks and n_sendable > 0:
        ks = [n_sendable]
    lift_table = [costs.budget_lift(result, k, mean_ev_all) for k in ks]

    return {
        "offer": {
            "category": offer["category"],
            "payout": offer["payout"],
            "commitment_level": offer["commitment_level"],
            "commitment_label": data.COMMITMENT_LADDER[offer["commitment_level"]],
            "channel": result["channel"],
        },
        "floor_per_lead": result["floor"],
        "n_leads_scored": len(leads),
        "n_sendable": n_sendable,
        "n_excluded": result["n_excluded"],
        "mean_ev_per_lead": round(mean_ev_all, 4),
        "top_sendable": result["sendable"][:5],
        "example_excluded": result["excluded"][:2],
        "budget_lift": lift_table,
    }


def tool_recommend_offers_for_lead(intent_category, source_platform=None,
                                   days_since_last_open=None, k=3) -> dict:
    lead = _sample_lead(intent_category)
    if source_platform:
        lead["source_platform"] = source_platform
    if days_since_last_open is not None:
        lead["days_since_last_open"] = int(days_since_last_open)
        # specifying recency implies at least one open — keep the lead consistent
        lead["has_opened"] = 1
        lead["total_opens"] = max(int(lead["total_opens"]), 1)

    model = get_model()
    offers = data.get_offers()
    recs = recommend(model, lead, offers, k=int(k) if k else 3)

    return {
        "lead": {
            "intent_category": lead["intent_category"],
            "source_platform": lead["source_platform"],
            "device": lead["device"],
            "geo_state": lead["geo_state"],
            "days_since_last_open": int(lead["days_since_last_open"]),
            "total_opens": int(lead["total_opens"]),
            "has_opened": int(lead["has_opened"]),
        },
        "recommendations": recs.to_dict("records"),
    }


def tool_draft_message(category, payout, commitment_level=None,
                       intent_category=None, offer_name=None) -> dict:
    offer = _build_offer(category, payout, commitment_level, offer_name=offer_name)
    lead = _sample_lead(intent_category or category)
    model = get_model()

    scored = costs._score_lead_for_offer(model, lead, offer)
    prob, ev, match = scored["prob_convert"], scored["expected_value"], scored["match"]
    copy = generate_copy(lead, offer, match, prob, ev)

    return {
        "offer": {
            "offer_name": offer["offer_name"],
            "category": offer["category"],
            "payout": offer["payout"],
            "commitment_level": offer["commitment_level"],
        },
        "lead": {
            "intent_category": lead["intent_category"],
            "opted_into": lead["opted_into"],
            "source_platform": lead["source_platform"],
            "days_since_last_open": int(lead["days_since_last_open"]),
        },
        "match": match,
        "prob_convert": round(prob, 4),
        "expected_value": round(ev, 2),
        "copy": copy,
    }


_TOOL_FNS = {
    "list_offers": tool_list_offers,
    "estimate_campaign": tool_estimate_campaign,
    "recommend_offers_for_lead": tool_recommend_offers_for_lead,
    "draft_message": tool_draft_message,
}


TOOLS = [
    {
        "name": "list_offers",
        "description": "List the synthetic offer catalog with per-category payout "
                       "ranges and commitment levels. Use to resolve an offer the "
                       "user names by title, or to ground the assumptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string",
                             "enum": data.CATEGORIES,
                             "description": "Optional category filter."},
            },
        },
    },
    {
        "name": "estimate_campaign",
        "description": "Estimate which leads to send an offer to and the projected "
                       "revenue/lift. Works for any described offer, even one not in "
                       "the catalog. payout only scales expected value; "
                       "commitment_level DOES affect the prediction — if the user "
                       "didn't state it, infer it and say which level you assumed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": data.CATEGORIES},
                "payout": {"type": "number",
                           "description": "Affiliate payout in USD per conversion."},
                "commitment_level": {
                    "type": "integer", "minimum": 1, "maximum": 5,
                    "description": "1=email_submit, 2=quote_request, 3=long_form, "
                                   "4=paid_trial, 5=purchase. Defaults to 2 if unknown."},
                "channel": {"type": "string", "enum": ["email", "sms"],
                            "description": "Send channel (default email)."},
                "min_ev": {"type": "number",
                           "description": "Optional per-lead expected-value floor in USD."},
                "budget_k": {"type": "integer",
                             "description": "Optional send-budget size for the lift comparison."},
            },
            "required": ["category", "payout"],
        },
    },
    {
        "name": "recommend_offers_for_lead",
        "description": "Given a described lead, rank the catalog offers by expected "
                       "value (top-k). Builds a representative lead from the given "
                       "traits; unspecified traits come from a typical lead of that intent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_category": {"type": "string", "enum": data.CATEGORIES},
                "source_platform": {"type": "string",
                                    "description": "e.g. google_search, meta, tiktok, taboola."},
                "days_since_last_open": {"type": "integer",
                                         "description": "Recency; higher = more dormant."},
                "k": {"type": "integer", "description": "How many offers to return (default 3)."},
            },
            "required": ["intent_category"],
        },
    },
    {
        "name": "draft_message",
        "description": "Draft email subject/body + an SMS for a representative lead "
                       "and a given offer. The copy is framed around the OFFER (pass "
                       "offer_name for the specific product, e.g. 'auto insurance'); the "
                       "lead only sets tone. Uses the copy model; it never re-ranks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": data.CATEGORIES},
                "payout": {"type": "number"},
                "commitment_level": {"type": "integer", "minimum": 1, "maximum": 5},
                "offer_name": {"type": "string",
                               "description": "The specific product the user described, "
                                              "e.g. 'auto insurance quote', 'SafeAuto auto "
                                              "policy'. Pass it through verbatim so the copy "
                                              "is framed around the real product, not the "
                                              "generic category."},
                "intent_category": {"type": "string", "enum": data.CATEGORIES,
                                    "description": "Intent of the representative lead; "
                                                   "defaults to the offer category."},
            },
            "required": ["category", "payout"],
        },
    },
]


def _run_tool(name: str, args: dict):
    fn = _TOOL_FNS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_agent(messages: list) -> dict:
    """Run the tool-use loop. `messages` is plain [{role, content: str}, ...].

    Returns {"reply": str, "messages": [...plain history + assistant reply]}.
    """
    client = _get_client()
    history = [{"role": m["role"], "content": m["content"]} for m in messages]
    working = list(history)

    for _ in range(MAX_ITERS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=working,
        )

        if resp.stop_reason == "tool_use":
            working.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = _run_tool(block.name, block.input or {})
                    tr = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out, default=str),
                    }
                    if isinstance(out, dict) and out.get("error"):
                        tr["is_error"] = True
                    tool_results.append(tr)
            working.append({"role": "user", "content": tool_results})
            continue

        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "messages": history}

    reply = ("I couldn't finish that within a reasonable number of steps — "
             "try rephrasing or breaking it into smaller questions.")
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply, "messages": history}
