"""
The campaign/send ledger — the simulated ESP's persistent outbox.

SQLite (stdlib, one file) rather than JSONL because the staged -> dispatched
transition is an in-place state update and, under gunicorn, two workers each
own an MCP server subprocess writing the same ledger. WAL mode + a conditional
UPDATE give cross-process locking and a dispatch-once guarantee for free.

The `sends.recipient` column holds the REAL (synthetic) email/phone joined from
leads.csv. It exists only here — every tool result that leaves the MCP server
masks contact info, so the LLM never sees a full address.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("CAMPAIGN_DB_PATH", os.path.join(BASE_DIR, "campaigns.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id       TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    dispatched_at     TEXT,
    status            TEXT NOT NULL,          -- 'staged' | 'dispatched'
    offer_name        TEXT,
    category          TEXT,
    payout            REAL,
    commitment_level  INTEGER,
    channel           TEXT,                    -- 'email' | 'sms'
    min_ev            REAL,
    budget_k          INTEGER,
    n_recipients      INTEGER,
    projected_revenue REAL,
    mode              TEXT,                    -- 'simulated' | 'live_sandbox'
    email_subject     TEXT,
    email_body        TEXT,
    sms_text          TEXT
);
CREATE TABLE IF NOT EXISTS sends (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id    TEXT NOT NULL REFERENCES campaigns(campaign_id),
    lead_id        TEXT NOT NULL,
    recipient      TEXT,                       -- real contact; never leaves the ledger unmasked
    redirected_to  TEXT,                       -- DEMO_RECIPIENT_* when sent live
    channel        TEXT,
    status         TEXT NOT NULL,              -- 'staged' | 'sent_simulated' | 'sent_live' | 'failed'
    error          TEXT,
    sent_at        TEXT,
    prob_convert   REAL,
    expected_value REAL
);
CREATE INDEX IF NOT EXISTS idx_sends_campaign ON sends(campaign_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def create_campaign(campaign: dict, sends: list) -> str:
    """Insert a staged campaign + its send rows in one transaction."""
    campaign_id = "cmp_" + uuid.uuid4().hex[:8]
    with _conn() as conn:
        conn.execute(
            """INSERT INTO campaigns (campaign_id, created_at, status, offer_name,
                   category, payout, commitment_level, channel, min_ev, budget_k,
                   n_recipients, projected_revenue, mode, email_subject, email_body,
                   sms_text)
               VALUES (?, ?, 'staged', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (campaign_id, _now(), campaign.get("offer_name"),
             campaign.get("category"), campaign.get("payout"),
             campaign.get("commitment_level"), campaign.get("channel"),
             campaign.get("min_ev"), campaign.get("budget_k"),
             len(sends), campaign.get("projected_revenue"), campaign.get("mode"),
             campaign.get("email_subject"), campaign.get("email_body"),
             campaign.get("sms_text")))
        conn.executemany(
            """INSERT INTO sends (campaign_id, lead_id, recipient, channel, status,
                   prob_convert, expected_value)
               VALUES (?, ?, ?, ?, 'staged', ?, ?)""",
            [(campaign_id, s["lead_id"], s["recipient"], campaign.get("channel"),
              s["prob_convert"], s["expected_value"]) for s in sends])
    return campaign_id


def get_campaign(campaign_id: str) -> dict | None:
    """Campaign row + per-status send counts, or None if unknown."""
    with _conn() as conn:
        row = conn.execute("SELECT * FROM campaigns WHERE campaign_id = ?",
                           (campaign_id,)).fetchone()
        if row is None:
            return None
        counts = conn.execute(
            "SELECT status, COUNT(*) AS n FROM sends WHERE campaign_id = ? "
            "GROUP BY status", (campaign_id,)).fetchall()
    out = dict(row)
    out["send_counts"] = {c["status"]: c["n"] for c in counts}
    return out


def list_campaigns(limit: int = 10) -> list:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT campaign_id, created_at, dispatched_at, status, offer_name,
                      category, channel, n_recipients, projected_revenue, mode
               FROM campaigns ORDER BY created_at DESC LIMIT ?""",
            (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def claim_for_dispatch(campaign_id: str) -> dict | None:
    """Atomically flip staged -> dispatched. None means unknown OR already
    dispatched — the cross-process dispatch-once guard."""
    with _conn() as conn:
        cur = conn.execute(
            """UPDATE campaigns SET status = 'dispatched', dispatched_at = ?
               WHERE campaign_id = ? AND status = 'staged'""",
            (_now(), campaign_id))
        if cur.rowcount == 0:
            return None
    return get_campaign(campaign_id)


def get_sends(campaign_id: str) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sends WHERE campaign_id = ? ORDER BY expected_value DESC",
            (campaign_id,)).fetchall()
    return [dict(r) for r in rows]


def update_send(send_id: int, status: str, redirected_to: str | None = None,
                error: str | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            """UPDATE sends SET status = ?, redirected_to = ?, error = ?, sent_at = ?
               WHERE id = ?""",
            (status, redirected_to, error, _now(), send_id))
