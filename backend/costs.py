"""
Send-cost + threshold logic — the "who do we send this offer to" decision.

This is what stops the tool from blasting every lead. A send is worth making
only when its expected revenue clears the cost of sending. Costs, by how
KNOWABLE they are (this honesty matters):

  1. hard cost   — real per-message fee. email ~ $0.001, SMS ~ $0.01 (Twilio-ish).
                   Fully knowable. Makes channel choice an EV decision.
  2. fatigue     — NOT cleanly quantifiable in dollars (unsubscribe/spam-complaint
                   risk degrades sender reputation for the WHOLE list). Modeled
                   operationally as a frequency cap, which is what real operators
                   actually enforce — not a fabricated per-send dollar figure.
  3. opportunity — the EV of the best OTHER offer you'd send this lead in the same
                   slot. Computed from the model's own EV matrix (see recommend.py
                   when ranking offers per lead). Left to the agent layer to apply
                   across a full multi-offer allocation; here we focus on one offer.

The transpose insight: recommend.py fixes a LEAD and ranks offers.
This module fixes an OFFER and ranks leads. Same P*payout matrix, read by column.
"""

import pandas as pd
from recommend import MATCH, _score_leads_vectorized

# Hard per-send cost by channel (USD). Tune to real ESP/SMS pricing.
SEND_COST = {"email": 0.001, "sms": 0.01}


def _score_lead_for_offer(model, lead: dict, offer: dict) -> dict:
    """Score one lead against one fixed offer -> prob, EV."""
    match = MATCH[lead["intent_category"]][offer["category"]]
    feat = {
        "source_platform": lead["source_platform"],
        "intent_category": lead["intent_category"],
        "geo_state": lead["geo_state"],
        "device": lead["device"],
        "offer_category": offer["category"],
        "intent_match": match,
        "days_since_signup": lead["days_since_signup"],
        "total_opens": lead["total_opens"],
        "has_opened": lead["has_opened"],
        "days_since_last_open": lead["days_since_last_open"],
        "commitment_level": offer["commitment_level"],
    }
    prob = float(model.predict_proba(pd.DataFrame([feat]))[0, 1])
    return {"prob_convert": prob, "expected_value": prob * offer["payout"], "match": match}


def target_offer(model, offer: dict, leads: pd.DataFrame,
                 channel: str = "email",
                 min_ev: float | None = None,
                 freq_cap_ok=None) -> dict:
    """
    Given ONE offer, decide which leads are worth sending to.

    channel    : 'email' or 'sms' -> sets the hard cost floor.
    min_ev     : optional explicit EV floor (e.g. agent extracts "at least $3/lead").
                 If None, defaults to the hard send cost for the channel.
    freq_cap_ok: optional callable(lead_id)->bool, the frequency-cap check
                 (fatigue proxy). If None, all leads pass the cap.

    Returns sendable segment + excluded leads WITH REASONS + projected revenue
    and lift vs. blasting the whole list.
    """
    cost = SEND_COST[channel]
    floor = cost if min_ev is None else max(min_ev, cost)

    # Vectorized: one predict_proba over all leads for this fixed offer.
    probs, matches = _score_leads_vectorized(model, offer, leads)
    payout = offer["payout"]

    sendable, excluded = [], []

    for i, (_, lead) in enumerate(leads.iterrows()):
        prob = float(probs[i])
        ev = prob * payout

        row = {"lead_id": lead["lead_id"], "prob_convert": round(prob, 4),
               "expected_value": round(ev, 2), "match": matches.iloc[i]}

        if freq_cap_ok is not None and not freq_cap_ok(lead["lead_id"]):
            excluded.append({**row, "reason": "frequency cap reached"})
        elif ev < floor:
            excluded.append({**row, "reason": f"EV ${ev:.2f} below floor ${floor:.2f}"})
        else:
            sendable.append({**row, "net_value": round(ev - cost, 2)})

    sendable.sort(key=lambda r: r["expected_value"], reverse=True)
    targeted_revenue = sum(r["net_value"] for r in sendable)

    # Honest value metric lives in budget_lift() below: given a SEND BUDGET
    # (you can't blast — fatigue/reputation/caps forbid it), how much better is
    # spending it on the model's top-K leads vs. a random K? That is the real
    # lift — best USE of a limited budget. (With ~zero hard cost, raw EV is
    # maximized by blasting; the reason you don't is fatigue, modeled as the
    # freq cap, not this EV math.)
    return {
        "offer": offer["offer_name"],
        "channel": channel,
        "floor": round(floor, 4),
        "n_sendable": len(sendable),
        "n_excluded": len(excluded),
        "targeted_revenue": round(targeted_revenue, 2),
        "sendable": sendable,
        "excluded": excluded,
    }


def budget_lift(result: dict, budget_k: int, mean_ev_all: float) -> dict:
    """Given a send budget K, compare model's top-K EV vs a random-K baseline."""
    top = result["sendable"][:budget_k]
    model_rev = sum(r["expected_value"] for r in top)
    random_rev = mean_ev_all * budget_k        # random K leads ~ mean EV each
    return {
        "budget_k": budget_k,
        "model_topk_revenue": round(model_rev, 2),
        "random_k_revenue": round(random_rev, 2),
        "lift": round(model_rev - random_rev, 2),
        "lift_multiple": round(model_rev / random_rev, 2) if random_rev else None,
    }


if __name__ == "__main__":
    from recommend import train_model

    model = train_model()
    leads = pd.read_csv("leads.csv")
    offers = pd.read_csv("offers.csv")

    offer = offers[offers.offer_id == "O003"].iloc[0].to_dict()  # SafeAuto Quote
    print(f"NEW OFFER: {offer['offer_name']} ({offer['category']}, "
          f"{offer['commitment_label']}, payout ${offer['payout']})\n")

    # Score everyone (floor at hard cost so nothing is dropped on EV alone here)
    r = target_offer(model, offer, leads, channel="email")
    mean_ev_all = sum(x["expected_value"] for x in r["sendable"]) / len(r["sendable"])

    print("The send decision is budget-constrained (can't blast — fatigue/caps).")
    print("Given a send budget K, spend it on the model's top-K leads vs a random K:\n")
    print(f"{'budget K':>10}{'model top-K $':>16}{'random K $':>14}{'lift $':>12}{'x':>7}")
    for k in [100, 300, 500, 1000]:
        bl = budget_lift(r, k, mean_ev_all)
        print(f"{k:>10}{bl['model_topk_revenue']:>16}{bl['random_k_revenue']:>14}"
              f"{bl['lift']:>12}{bl['lift_multiple']:>7}")

    print(f"\nWith an explicit business floor (min_ev=$8), who clears the bar:")
    r8 = target_offer(model, offer, leads, channel="email", min_ev=8)
    print(f"  sendable: {r8['n_sendable']}/{len(leads)}   excluded: {r8['n_excluded']}")
    print(f"  sample excluded reason: {r8['excluded'][0]['reason'] if r8['excluded'] else 'none'}")
