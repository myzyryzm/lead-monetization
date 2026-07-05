"""
LLM layer: turn a model-ranked (lead, offer) pair into operator-ready output —
a plain-English rationale plus tailored email + SMS copy.

Division of labor (the whole architecture in one sentence):
  the MODEL decides WHICH offers and HOW valuable (prob, EV, ranking);
  the LLM only EXPLAINS and WRITES — it never re-ranks.

Requires ANTHROPIC_API_KEY in the environment. Falls back to a rule-based
stub if the key is missing or the call fails, so the tool still runs offline.
"""

import os
import json
import pandas as pd
from anthropic import Anthropic

MODEL = "claude-haiku-4-5"   # fast + cheap: copy gen runs at list scale

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _client


def _fallback(lead, offer, match):
    """Used when no API key / call fails, so the demo never hard-breaks."""
    return {
        "rationale": f"{match}: lead intent '{lead['intent_category']}' "
                     f"vs offer category '{offer['category']}'.",
        "email_subject": f"An offer picked for you: {offer['offer_name']}",
        "email_body": f"Hi — based on your interests we think {offer['offer_name']} "
                      f"could be a strong fit. Take a look.",
        "sms": f"{offer['offer_name']} — picked for you. Tap to see details.",
    }


def generate_copy(lead: dict, offer: dict, match: str,
                  prob: float, ev: float) -> dict:
    """Ask the LLM for rationale + email + SMS as structured JSON."""
    prompt = f"""You are a marketing copy assistant for an affiliate company. \
A conversion model has already RANKED this offer for this lead. Do NOT \
question the ranking — your job is to (1) explain the pick in one crisp \
sentence a media buyer would respect, and (2) write tailored outreach copy.

LEAD
- original interest (opted into): {lead['opted_into']}
- intent category: {lead['intent_category']}
- acquisition source: {lead['source_platform']}
- days since last engaged: {"never engaged" if not lead.get('has_opened', 1) else lead['days_since_last_open']}

OFFER
- name: {offer['offer_name']}
- category: {offer['category']}
- payout to us: ${offer['payout']}

MODEL OUTPUT
- intent match level: {match}
- predicted conversion probability: {prob:.1%}
- expected value: ${ev:.2f}

Write copy that fits the lead's original interest and re-engages them if \
they've gone quiet. Keep email body under 60 words, SMS under 160 chars.

Respond with ONLY a JSON object, no markdown, no preamble:
{{"rationale": "...", "email_subject": "...", "email_body": "...", "sms": "..."}}"""

    try:
        resp = _get_client().messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        # strip accidental code fences before parsing
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[llm fallback: {type(e).__name__}: {e}]")
        return _fallback(lead, offer, match)


if __name__ == "__main__":
    from recommend import train_model, recommend, MATCH

    model = train_model()
    leads = pd.read_csv("leads.csv")
    offers = pd.read_csv("offers.csv")

    lead = leads[leads.intent_category == "finance"].iloc[0].to_dict()
    recs = recommend(model, lead, offers, k=2)

    print(f"LEAD: opted_into={lead['opted_into']} source={lead['source_platform']} "
          f"days_since_last_open={lead['days_since_last_open']}\n")

    for _, row in recs.iterrows():
        offer = offers[offers.offer_id == row["offer_id"]].iloc[0].to_dict()
        match = MATCH[lead["intent_category"]][offer["category"]]
        out = generate_copy(lead, offer, match,
                            row["prob_convert"], row["expected_value"])
        print("=" * 64)
        print(f"{row['offer_name']}  | P={row['prob_convert']:.1%}  EV=${row['expected_value']}")
        print("-" * 64)
        print("rationale :", out["rationale"])
        print("subject   :", out["email_subject"])
        print("email     :", out["email_body"])
        print("sms       :", out["sms"])
        print()
