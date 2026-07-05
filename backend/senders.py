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


def send_sms_live(text: str, intended_lead_id: str) -> str:
    """Send one real SMS via Twilio — always to DEMO_RECIPIENT_PHONE. Raises on failure."""
    sid = os.environ["TWILIO_ACCOUNT_SID"]
    demo = os.environ["DEMO_RECIPIENT_PHONE"]

    resp = requests.post(
        _TWILIO_URL.format(sid=sid),
        auth=(sid, os.environ["TWILIO_AUTH_TOKEN"]),
        data={"From": os.environ["TWILIO_FROM_NUMBER"], "To": demo,
              "Body": f"[DEMO {intended_lead_id}] {text}"[:1600]},
        timeout=30)
    resp.raise_for_status()
    return demo


def mask_email(addr: str) -> str:
    local, _, domain = str(addr).partition("@")
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"


def mask_phone(num: str) -> str:
    num = str(num)
    return f"{num[:5]}***{num[-4:]}" if len(num) > 9 else "***"
