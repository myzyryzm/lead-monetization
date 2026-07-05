"""
The campaign-dispatch MCP server — the "ESP side" of the system.

Runs as a stdio subprocess of the backend (spawned by mcp_client.py) and
exposes four tools: stage_campaign, dispatch_campaign, get_campaign_status,
list_campaigns. It owns everything the agent must NOT: the outbox ledger, the
lead_id -> email/phone join, and the SMTP/Twilio credentials. Every tool result
masks contact info, so the LLM never sees a full address — and ANTHROPIC_API_KEY
is never forwarded to this process (see mcp_client._PASS_ENV).

stdout is the JSON-RPC transport, so nothing here may print: model_store logs
to stdout on retrain, hence the redirect during startup imports.
"""

import contextlib
import os
import sys

with contextlib.redirect_stdout(sys.stderr):
    import data
    import costs
    import ledger
    import senders
    from model_store import get_model

    data.warm()
    _MODEL = get_model()
    ledger.init_db()
    # lead_id -> contact info; lives only in this process.
    _CONTACTS = data.get_leads().set_index("lead_id")[["email", "phone"]]

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("campaign-dispatch")


def _mask(contact: str, channel: str) -> str:
    return (senders.mask_email(contact) if channel == "email"
            else senders.mask_phone(contact))


@mcp.tool()
def stage_campaign(category: str, payout: float, channel: str = "email",
                   email_subject: str = None, email_body: str = None,
                   sms_text: str = None, commitment_level: int = 2,
                   min_ev: float = None, budget_k: int = None,
                   lead_ids: list = None, offer_name: str = None) -> dict:
    """Stage (but do NOT send) an outreach campaign: score all leads for the
    offer, resolve the target segment, and record it with the provided copy.
    Draft the copy first (draft_message) and pass it in. Returns a campaign_id
    and a summary to present to the user for approval — nothing is sent until
    dispatch_campaign is called."""
    if channel not in ("email", "sms"):
        return {"error": f"unknown channel: {channel}"}
    if channel == "email" and not (email_subject and email_body):
        return {"error": "email campaigns need email_subject and email_body "
                         "(draft them with draft_message first)"}
    if channel == "sms" and not sms_text:
        return {"error": "sms campaigns need sms_text (draft it with "
                         "draft_message first)"}

    leads = data.get_leads()
    if lead_ids:
        leads = leads[leads.lead_id.isin(lead_ids)]
        if len(leads) == 0:
            return {"error": "none of the given lead_ids exist"}

    offer = {
        "offer_name": offer_name or f"{category.replace('_', ' ').title()} offer",
        "category": category,
        "commitment_level": int(commitment_level or 2),
        "payout": float(payout),
    }
    result = costs.target_offer(_MODEL, offer, leads, channel=channel,
                                min_ev=min_ev)

    recipients = result["sendable"]
    if budget_k:
        recipients = recipients[:int(budget_k)]
    if not recipients:
        return {"error": "no leads clear the EV floor for this offer — "
                         "nothing to stage"}

    sends = []
    for r in recipients:
        lead_id = str(r["lead_id"])
        contact = _CONTACTS.loc[lead_id, "email" if channel == "email" else "phone"]
        sends.append({"lead_id": lead_id, "recipient": str(contact),
                      "prob_convert": float(r["prob_convert"]),
                      "expected_value": float(r["expected_value"])})

    projected_revenue = round(sum(s["expected_value"] for s in sends), 2)
    mode = "live_sandbox" if senders.live_mode(channel) else "simulated"

    campaign_id = ledger.create_campaign(
        {"offer_name": offer["offer_name"], "category": category,
         "payout": float(payout), "commitment_level": offer["commitment_level"],
         "channel": channel, "min_ev": float(min_ev) if min_ev else None,
         "budget_k": int(budget_k) if budget_k else None,
         "projected_revenue": projected_revenue, "mode": mode,
         "email_subject": email_subject, "email_body": email_body,
         "sms_text": sms_text},
        sends)

    if mode == "live_sandbox":
        demo = os.environ.get("DEMO_RECIPIENT_EMAIL" if channel == "email"
                              else "DEMO_RECIPIENT_PHONE", "")
        live_send_plan = {
            "would_send_live": min(senders.live_send_cap(), len(sends)),
            "redirect_to": _mask(demo, channel),
            "note": "live sends are sandbox-redirected to the operator's own "
                    "inbox/phone; the remainder are simulated"}
    else:
        live_send_plan = ("all sends simulated (LIVE_SEND off or credentials "
                          "missing for this channel)")

    return {
        "campaign_id": campaign_id,
        "status": "staged",
        "mode": mode,
        "channel": channel,
        "offer_name": offer["offer_name"],
        "n_recipients": len(sends),
        "n_excluded": int(result["n_excluded"]),
        "ev_floor": float(result["floor"]),
        "projected_revenue": projected_revenue,
        "live_send_plan": live_send_plan,
        "copy_preview": {"email_subject": email_subject,
                         "email_body": email_body, "sms_text": sms_text},
        "top_recipients": [
            {"lead_id": s["lead_id"], "contact": _mask(s["recipient"], channel),
             "prob_convert": s["prob_convert"],
             "expected_value": s["expected_value"]}
            for s in sends[:5]],
        "next_step": "Present this summary to the user and WAIT for their "
                     "explicit approval before calling dispatch_campaign.",
    }


@mcp.tool()
def dispatch_campaign(campaign_id: str) -> dict:
    """Execute a staged campaign. ONLY call this after the user has explicitly
    approved the staged summary in the conversation. Each campaign can be
    dispatched exactly once."""
    camp = ledger.claim_for_dispatch(campaign_id)
    if camp is None:
        return {"error": f"{campaign_id} is unknown or already dispatched "
                         "(check get_campaign_status)"}

    channel = camp["channel"]
    live = camp["mode"] == "live_sandbox"
    cap = senders.live_send_cap() if live else 0

    n_live = n_sim = n_failed = 0
    live_sends, first_error = [], None
    for i, send in enumerate(ledger.get_sends(campaign_id)):   # EV-descending
        if i < cap:
            try:
                if channel == "email":
                    demo = senders.send_email_live(
                        camp["email_subject"], camp["email_body"], send["lead_id"])
                else:
                    demo = senders.send_sms_live(camp["sms_text"], send["lead_id"])
                ledger.update_send(send["id"], "sent_live", redirected_to=demo)
                live_sends.append({"lead_id": send["lead_id"],
                                   "redirected_to": _mask(demo, channel)})
                n_live += 1
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                ledger.update_send(send["id"], "failed", error=err)
                first_error = first_error or err
                n_failed += 1
        else:
            ledger.update_send(send["id"], "sent_simulated")
            n_sim += 1

    out = {"campaign_id": campaign_id, "status": "dispatched",
           "mode": camp["mode"], "channel": channel,
           "n_sent_live": n_live, "n_sent_simulated": n_sim,
           "n_failed": n_failed, "live_sends": live_sends}
    if first_error:
        out["first_error"] = first_error
    return out


@mcp.tool()
def get_campaign_status(campaign_id: str) -> dict:
    """Status of one campaign: staged/dispatched, mode, recipient count,
    projected revenue, and per-send-status counts."""
    camp = ledger.get_campaign(campaign_id)
    if camp is None:
        return {"error": f"unknown campaign: {campaign_id}"}
    camp.pop("email_body", None)   # keep results compact
    return camp


@mcp.tool()
def list_campaigns(limit: int = 10) -> dict:
    """List recent campaigns (newest first)."""
    return {"campaigns": ledger.list_campaigns(limit)}


if __name__ == "__main__":
    mcp.run()   # stdio transport
