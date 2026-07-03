"""
Synthetic data generator for the lead-monetization model.

Produces three tables:
  leads.csv   - one row per person (the user who opted in)
  offers.csv  - one row per affiliate offer we can send
  pairs.csv   - one row per (lead, offer) -> the model's training data

The hidden conversion-probability function is the invented "truth".
The model never sees true_p; it only sees features -> converted (0/1).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # reproducible

# ----------------------------------------------------------------------
# 1. TAXONOMY
# ----------------------------------------------------------------------
CATEGORIES = ["finance", "insurance", "health", "sweepstakes", "home_services"]

# High-cardinality opt-in strings -> their intent_category.
# (In production an LLM would map free-text opt-ins into these buckets.)
OPT_INS = {
    "finance":       ["mortgage_refi_quiz", "debt_relief_guide", "personal_loan_calc"],
    "insurance":     ["auto_insurance_quiz", "life_insurance_quote", "health_ins_compare"],
    "health":        ["weight_loss_guide", "keto_meal_plan", "supplement_trial"],
    "sweepstakes":   ["500_giftcard_sweeps", "win_iphone_giveaway", "weekly_cash_draw"],
    "home_services": ["solar_savings_quote", "roofing_estimate", "hvac_tuneup_offer"],
}

# 5x5 match matrix: lead intent (row) x offer category (col)
MATCH = {
    "finance":       {"finance":"match","insurance":"related","health":"mismatch","sweepstakes":"related","home_services":"related"},
    "insurance":     {"finance":"related","insurance":"match","health":"mismatch","sweepstakes":"related","home_services":"related"},
    "health":        {"finance":"mismatch","insurance":"mismatch","health":"match","sweepstakes":"related","home_services":"mismatch"},
    "sweepstakes":   {"finance":"mismatch","insurance":"mismatch","health":"mismatch","sweepstakes":"match","home_services":"mismatch"},
    "home_services": {"finance":"related","insurance":"related","health":"mismatch","sweepstakes":"related","home_services":"match"},
}

# Payout tiers by category (sweeps cheap, finance/insurance rich)
PAYOUT_RANGE = {
    "finance":       (40, 120),
    "insurance":     (35, 90),
    "health":        (15, 45),
    "sweepstakes":   (1, 8),
    "home_services": (30, 100),
}

# commitment_level = what the LEAD must DO for the conversion to fire.
# Ordinal: higher number = more effort/risk = lower conversion (the causal
# driver that replaces payout in the conversion model). Payout stays ONLY
# in the EV multiplier, since the consumer never sees it.
COMMITMENT_LEVELS = {
    1: "email_submit",   # hand over email/zip — frictionless
    2: "quote_request",  # short form to "see your quote"
    3: "long_form",      # multi-step application
    4: "paid_trial",     # enter payment, start trial
    5: "purchase",       # buy outright
}

# commitment is drawn CONDITIONAL on category (not uniform): sweeps are
# almost always frictionless email-grabs; health skews to paid actions; etc.
COMMITMENT_BY_CATEGORY = {
    "finance":       [2, 3, 3],        # quote / application heavy
    "insurance":     [2, 2, 3],        # mostly quote requests
    "health":        [1, 4, 5],        # free guide OR buy the supplement
    "sweepstakes":   [1, 1, 1],        # always frictionless entry
    "home_services": [2, 3],           # quote / estimate
}

# Effect of commitment on TRUE conversion prob. Mostly monotonic (more
# effort -> lower conversion), with one deliberate non-monotonic kink:
# paid_trial (4) converts slightly BETTER than long_form (3), because a
# tedious application loses more people than a quick card entry from
# someone already decided. Lets us later test ordinal vs one-hot encoding.
COMMITMENT_EFFECT = {
    1:  0.06,   # email_submit  — easiest, highest lift
    2:  0.02,   # quote_request
    3: -0.03,   # long_form     — tedious, big dropoff
    4: -0.01,   # paid_trial    — kink: better than long_form
    5: -0.05,   # purchase      — hardest
}

PLATFORMS = ["google_search", "meta", "tiktok", "taboola"]
PLATFORM_WEIGHTS = [0.20, 0.35, 0.30, 0.15]   # most volume from meta/tiktok
SOURCE_QUALITY = {"google_search":0.04, "meta":0.02, "tiktok":0.00, "taboola":-0.01}
DEVICES = ["mobile", "desktop", "tablet"]
STATES = ["TX","CA","FL","NY","OH","GA","NC","PA","IL","AZ"]

# ----------------------------------------------------------------------
# 2. GENERATE LEADS
# ----------------------------------------------------------------------
def make_leads(n):
    rows = []
    for i in range(n):
        cat = rng.choice(CATEGORIES)
        opted = rng.choice(OPT_INS[cat])
        platform = rng.choice(PLATFORMS, p=PLATFORM_WEIGHTS)
        days_since_signup = int(rng.integers(0, 365))
        # engagement: most leads are lukewarm-to-cold (realistic)
        total_opens = int(rng.poisson(2))
        # dormancy correlates loosely with signup age
        base_dormancy = rng.integers(0, 120)
        days_since_last_open = int(min(base_dormancy + days_since_signup // 4, 365))
        rows.append({
            "lead_id": f"L{i:05d}",
            "email": f"user{i}@example.com",
            "phone": f"+1555{i:07d}",
            "signup_ts": (pd.Timestamp("2026-06-29") - pd.Timedelta(days=days_since_signup)).date(),
            "source_platform": platform,
            "campaign_id": f"camp_{platform}_{rng.integers(1,9)}",
            "ad_id": f"ad_{rng.integers(1000,9999)}",
            "opted_into": opted,
            "intent_category": cat,
            "geo_state": rng.choice(STATES),
            "device": rng.choice(DEVICES),
            "days_since_signup": days_since_signup,
            "total_opens": total_opens,
            "days_since_last_open": days_since_last_open,
        })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 3. GENERATE OFFERS
# ----------------------------------------------------------------------
def make_offers():
    rows = []
    oid = 0
    names = {
        "finance":["RefiPro Mortgage","QuickCash Loan","CreditFix Plus"],
        "insurance":["SafeAuto Quote","LifeShield Term","HomeGuard Warranty"],
        "health":["LeanFast Supplement","TeleDoc Trial","KetoBox Plan"],
        "sweepstakes":["MegaCash Draw","GiftCard Spin","DailyWin Survey"],
        "home_services":["SolarSave Install","RoofRight Estimate","CoolAir HVAC"],
    }
    for cat in CATEGORIES:
        lo, hi = PAYOUT_RANGE[cat]
        for nm in names[cat]:
            level = int(rng.choice(COMMITMENT_BY_CATEGORY[cat]))
            rows.append({
                "offer_id": f"O{oid:03d}",
                "offer_name": nm,
                "category": cat,
                "commitment_level": level,
                "commitment_label": COMMITMENT_LEVELS[level],
                "payout": round(float(rng.uniform(lo, hi)), 2),
                "eligibility": "US, 18+",   # plain text -> LLM parses later
            })
            oid += 1
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
# 4. HIDDEN TRUTH FUNCTION  (model never sees this)
# ----------------------------------------------------------------------
def true_prob(lead, offer):
    p = 0.02
    m = MATCH[lead["intent_category"]][offer["category"]]
    p += {"match":0.12, "related":0.05, "mismatch":0.0}[m]

    d = lead["days_since_last_open"]
    if d <= 7: p += 0.05
    elif d <= 30: p += 0.02
    elif d > 90: p -= 0.02
    p += min(lead["total_opens"], 10) * 0.003

    p += SOURCE_QUALITY[lead["source_platform"]]

    s = lead["days_since_signup"]
    if s <= 14: p += 0.03
    elif s > 180: p -= 0.02

    # commitment_level drives conversion (causal: more effort -> fewer convert).
    # This REPLACES the old payout term — payout is invisible to the consumer
    # and belongs only in EV = P * payout, not in P itself.
    p += COMMITMENT_EFFECT[offer["commitment_level"]]

    p += rng.normal(0, 0.02)
    return float(np.clip(p, 0.001, 0.95))

# ----------------------------------------------------------------------
# 5. BUILD PAIRS  (lead x offer cross product -> training rows)
# ----------------------------------------------------------------------
def make_pairs(leads, offers):
    rows = []
    for _, lead in leads.iterrows():
        for _, offer in offers.iterrows():
            tp = true_prob(lead, offer)
            converted = int(rng.random() < tp)
            rows.append({
                "lead_id": lead["lead_id"],
                "offer_id": offer["offer_id"],
                # lead features
                "source_platform": lead["source_platform"],
                "intent_category": lead["intent_category"],
                "geo_state": lead["geo_state"],
                "device": lead["device"],
                "days_since_signup": lead["days_since_signup"],
                "total_opens": lead["total_opens"],
                "days_since_last_open": lead["days_since_last_open"],
                # offer features
                "offer_category": offer["category"],
                "commitment_level": offer["commitment_level"],
                # payout is kept for EV computation downstream, NOT a model feature
                "payout": offer["payout"],
                # engineered interaction feature
                "intent_match": MATCH[lead["intent_category"]][offer["category"]],
                # scaffolding + label
                "true_p": round(tp, 4),
                "converted": converted,
            })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    leads = make_leads(2000)
    offers = make_offers()
    pairs = make_pairs(leads, offers)

    leads.to_csv("leads.csv", index=False)
    offers.to_csv("offers.csv", index=False)
    pairs.to_csv("pairs.csv", index=False)

    print(f"leads:  {len(leads):>6}  rows")
    print(f"offers: {len(offers):>6}  rows")
    print(f"pairs:  {len(pairs):>6}  rows")
    print(f"\noverall conversion rate: {pairs['converted'].mean():.3%}")
    print("\nconversion rate by intent_match:")
    print(pairs.groupby("intent_match")["converted"].mean().to_string())
    print("\n--- sample offers ---")
    print(offers.to_string(index=False))
    print("\n--- sample pairs (5 rows) ---")
    print(pairs.head(5).to_string(index=False))
