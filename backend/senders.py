"""
Live-send transports with a hardcoded sandbox redirect.

Safety invariant: the send functions take NO recipient parameter. In live mode
every message goes to DEMO_RECIPIENT_EMAIL / DEMO_RECIPIENT_PHONE (the
operator's own inbox/phone) — the lead a message was *intended* for appears
only as a [DEMO ...] tag. Approving a 500-send campaign can therefore produce
at most LIVE_SEND_CAP (<= 3) real messages; the rest are simulated.

Live mode requires LIVE_SEND=1 AND the channel's full credential set. Anything
missing -> that channel silently stays simulated (the staged summary says so).
"""

import os
import smtplib
from email.message import EmailMessage

import requests

_TWILIO_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

_CHANNEL_VARS = {
    "email": ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "DEMO_RECIPIENT_EMAIL"),
    "sms": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
            "DEMO_RECIPIENT_PHONE"),
}


def live_mode(channel: str) -> bool:
    if os.environ.get("LIVE_SEND", "").lower() not in ("1", "true", "yes"):
        return False
    return all(os.environ.get(v) for v in _CHANNEL_VARS.get(channel, ("_missing",)))


def live_send_cap() -> int:
    try:
        cap = int(os.environ.get("LIVE_SEND_CAP", "1"))
    except ValueError:
        cap = 1
    return max(0, min(cap, 3))


def send_email_live(subject: str, body: str, intended_lead_id: str) -> str:
    """Send one real email — always to DEMO_RECIPIENT_EMAIL. Raises on failure."""
    sender = os.environ["GMAIL_ADDRESS"]
    demo = os.environ["DEMO_RECIPIENT_EMAIL"]

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = demo
    msg["Subject"] = f"[DEMO → {intended_lead_id}] {subject}"
    msg.set_content(
        f"{body}\n\n--\nSandbox redirect: this send was intended for lead "
        f"{intended_lead_id} and rerouted to you for the demo.")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(sender, os.environ["GMAIL_APP_PASSWORD"])
        smtp.send_message(msg)
    return demo


def _post_twilio(body: str):
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    return requests.post(
        _TWILIO_URL.format(sid=sid),
        auth=(sid, os.environ["TWILIO_AUTH_TOKEN"]),
        data={"From": os.environ["TWILIO_FROM_NUMBER"],
              "To": os.environ["DEMO_RECIPIENT_PHONE"],
              "Body": body[:1600]},
        timeout=30)


def _twilio_error(resp) -> tuple:
    try:
        err = resp.json()
        return err.get("code"), f"Twilio error {err.get('code')}: {err.get('message')}"
    except ValueError:
        return None, resp.text[:300]


def send_sms_live(text: str, intended_lead_id: str) -> str:
    """Send one real SMS via Twilio — always to DEMO_RECIPIENT_PHONE. Raises on failure."""
    demo = os.environ["DEMO_RECIPIENT_PHONE"]

    resp = _post_twilio(f"[DEMO {intended_lead_id}] {text}")
    code, detail = (None, None) if resp.ok else _twilio_error(resp)
    if code == 572006:
        # New-style Twilio trial: the body must be a predefined template NAME
        # (Twilio substitutes its canned text). The drafted copy can't go over
        # the wire until the account is upgraded — it still lives in the ledger.
        resp = _post_twilio(os.environ.get("TWILIO_TRIAL_TEMPLATE",
                                           "sms_marketing_promotions"))
        if not resp.ok:
            code, detail = _twilio_error(resp)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code} — {detail}")
    return demo


def mask_email(addr: str) -> str:
    local, _, domain = str(addr).partition("@")
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"


def mask_phone(num: str) -> str:
    num = str(num)
    return f"{num[:5]}***{num[-4:]}" if len(num) > 9 else "***"
