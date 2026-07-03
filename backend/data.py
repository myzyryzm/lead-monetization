"""
Absolute-path data access + catalog summary.

The engine scripts read the CSVs by relative path (fine when you run them from
`backend/`). The Flask app may run from a different working directory, so this
module is the single place that loads the data by ABSOLUTE path and caches it.
`catalog_summary()` is the one source of truth shared by the `/api/assumptions`
endpoint and the agent's `list_offers` tool.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS_PATH = os.path.join(BASE_DIR, "leads.csv")
OFFERS_PATH = os.path.join(BASE_DIR, "offers.csv")

CATEGORIES = ["finance", "insurance", "health", "sweepstakes", "home_services"]

# Ordinal commitment ladder (effort the lead must expend). This IS a model
# feature, so the agent must state which level it assumed for a described offer.
COMMITMENT_LADDER = {
    1: "email_submit",
    2: "quote_request",
    3: "long_form",
    4: "paid_trial",
    5: "purchase",
}

_leads = None
_offers = None


def get_leads() -> pd.DataFrame:
    global _leads
    if _leads is None:
        _leads = pd.read_csv(LEADS_PATH)
    return _leads


def get_offers() -> pd.DataFrame:
    global _offers
    if _offers is None:
        _offers = pd.read_csv(OFFERS_PATH)
    return _offers


def warm() -> None:
    """Load both CSVs into the cache (called at app startup)."""
    get_leads()
    get_offers()


def catalog_summary() -> dict:
    """Per-category catalog stats + offers, computed from offers.csv/leads.csv."""
    offers = get_offers()
    leads = get_leads()

    categories = []
    for cat in CATEGORIES:
        rows = offers[offers.category == cat]
        categories.append({
            "category": cat,
            "count": int(len(rows)),
            "payout_min": round(float(rows.payout.min()), 2),
            "payout_max": round(float(rows.payout.max()), 2),
            "offers": [
                {
                    "offer_id": r.offer_id,
                    "offer_name": r.offer_name,
                    "commitment_level": int(r.commitment_level),
                    "commitment_label": r.commitment_label,
                    "payout": round(float(r.payout), 2),
                }
                for r in rows.itertuples(index=False)
            ],
        })

    return {
        "n_leads": int(len(leads)),
        "n_offers": int(len(offers)),
        "categories": categories,
        "commitment_ladder": COMMITMENT_LADDER,
    }
